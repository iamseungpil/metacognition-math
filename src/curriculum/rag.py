"""Meta-guided curriculum learning via RAG.

Builds targeted training curricula by:
1. Collecting AIME rollouts and extracting meta weakness diagnoses
2. Building a sentence embedding index from MATH + GSM8K training problems
3. Searching for similar problems for each weakness category
4. Combining weakness-targeted problems into a curriculum dataset

Dependencies (install before use):
    pip install sentence-transformers faiss-cpu

Usage:
    from src.curriculum.rag import CurriculumRAG

    rag = CurriculumRAG(embedding_model="all-MiniLM-L6-v2")
    rag.build_problem_index(["MATH", "gsm8k"])
    weaknesses = rag.extract_weaknesses(rollouts)
    curriculum = rag.build_curriculum(weaknesses, top_k=50)
    curriculum.save_to_disk("sft_data/curriculum.parquet")
"""
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """A single model rollout with parsed meta blocks."""

    problem: str
    response: str
    correct: bool
    meta_blocks: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class WeaknessDiagnosis:
    """Aggregated weakness diagnosis across multiple rollouts."""

    category: str
    description: str
    count: int
    example_problems: list[str] = field(default_factory=list)


@dataclass
class ProblemEntry:
    """A problem in the embedding index."""

    text: str
    source: str  # "MATH", "gsm8k", etc.
    subject: str  # "algebra", "geometry", etc.
    difficulty: str  # "easy", "medium", "hard"
    solution: Optional[str] = None


# ---------------------------------------------------------------------------
# Meta block parsing
# ---------------------------------------------------------------------------

META_BLOCK_PATTERN = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.DOTALL)

# Weakness categories mapped from common meta-block language
WEAKNESS_CATEGORIES = {
    "number_theory": [
        "number theory", "modular", "mod ", "gcd", "lcm", "prime",
        "divisor", "divisibility", "congruence", "euler", "fermat",
    ],
    "geometry": [
        "geometry", "triangle", "circle", "angle", "polygon",
        "coordinate", "area", "perimeter", "similar", "congruent",
        "circumscrib", "inscrib", "tangent",
    ],
    "algebra": [
        "algebra", "equation", "polynomial", "factor", "quadratic",
        "system of equations", "inequality", "expression", "variable",
        "substitut",
    ],
    "combinatorics": [
        "combinat", "counting", "permutation", "combination",
        "probability", "expected value", "binomial", "pascal",
        "inclusion-exclusion", "pigeonhole",
    ],
    "calculus": [
        "calculus", "derivative", "integral", "limit", "series",
        "convergence", "taylor", "differential",
    ],
    "algebraic_manipulation": [
        "sign error", "simplif", "manipulation", "expanding",
        "collect terms", "cancel", "rearrang",
    ],
    "strategy_selection": [
        "approach", "strategy", "method", "try another", "different way",
        "complicated", "cleaner",
    ],
    "arithmetic_error": [
        "arithmetic", "calculation error", "compute", "miscalculat",
        "wrong value", "double-check",
    ],
}


def parse_meta_blocks(text: str) -> list[str]:
    """Extract all meta block contents from a response."""
    return [m.strip() for m in META_BLOCK_PATTERN.findall(text)]


def classify_weakness(meta_content: str) -> list[str]:
    """Classify a meta block into weakness categories.

    Returns all matching categories (a block may indicate multiple weaknesses).
    """
    content_lower = meta_content.lower()
    matched = []
    for category, keywords in WEAKNESS_CATEGORIES.items():
        if any(kw in content_lower for kw in keywords):
            matched.append(category)
    return matched


# ---------------------------------------------------------------------------
# Core RAG class
# ---------------------------------------------------------------------------


class CurriculumRAG:
    """RAG-based curriculum builder for meta-guided learning.

    Collects weakness diagnoses from metacognitive rollouts, then retrieves
    similar training problems from a sentence-embedding index to build
    targeted curricula.

    Args:
        embedding_model: Name of the sentence-transformers model to use.
            Default "all-MiniLM-L6-v2" is fast and ~80MB.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._embedding_model_name = embedding_model
        self._encoder = None  # lazy init
        self._index = None  # faiss index
        self._problems: list[ProblemEntry] = []

    # ------------------------------------------------------------------
    # Lazy encoder loading
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        """Lazily load sentence-transformers model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CurriculumRAG. "
                    "Install with: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self._embedding_model_name)
        return self._encoder

    # ------------------------------------------------------------------
    # Step 1: Collect rollouts and parse meta blocks
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        model,
        tokenizer,
        problems: list[dict[str, Any]],
        n_rollouts: int = 8,
        max_new_tokens: int = 2048,
        batch_size: int = 4,
    ) -> list[Rollout]:
        """Generate rollouts and parse meta blocks.

        Args:
            model: HuggingFace causal LM (already on GPU).
            tokenizer: Corresponding tokenizer.
            problems: List of dicts with keys "question" and "answer".
            n_rollouts: Number of rollouts per problem.
            max_new_tokens: Max generation length.
            batch_size: Batch size for generation.

        Returns:
            List of Rollout objects with parsed meta blocks.
        """
        import torch

        rollouts = []
        model.eval()

        for prob in tqdm(problems, desc="Collecting rollouts"):
            question = prob["question"]
            answer = prob.get("answer", "")

            for _ in range(n_rollouts):
                messages = [{"role": "user", "content": question}]
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                response = tokenizer.decode(
                    output[0][input_ids.shape[1]:],
                    skip_special_tokens=False,
                )

                meta_blocks = parse_meta_blocks(response)
                # Simple correctness check: does the response contain the answer?
                correct = _check_answer(response, answer)

                # Classify weaknesses from meta blocks
                weaknesses = []
                for block in meta_blocks:
                    weaknesses.extend(classify_weakness(block))
                weaknesses = list(set(weaknesses))

                rollouts.append(Rollout(
                    problem=question,
                    response=response,
                    correct=correct,
                    meta_blocks=meta_blocks,
                    weaknesses=weaknesses,
                ))

        return rollouts

    # ------------------------------------------------------------------
    # Step 2: Extract weakness diagnoses
    # ------------------------------------------------------------------

    def extract_weaknesses(
        self,
        rollouts: list[Rollout],
        min_count: int = 3,
    ) -> list[WeaknessDiagnosis]:
        """Categorize weaknesses from meta content across rollouts.

        Only includes categories that appear in at least `min_count` incorrect
        rollouts, to avoid noise from one-off errors.

        Args:
            rollouts: List of Rollout objects from collect_rollouts().
            min_count: Minimum number of incorrect rollouts for a category
                to be considered a real weakness.

        Returns:
            List of WeaknessDiagnosis objects sorted by frequency (desc).
        """
        # Only look at incorrect rollouts for weakness diagnosis
        incorrect = [r for r in rollouts if not r.correct]
        if not incorrect:
            print("Warning: No incorrect rollouts found. Model may be too strong for these problems.")
            return []

        category_counts: Counter = Counter()
        category_examples: dict[str, list[str]] = defaultdict(list)

        for rollout in incorrect:
            for weakness in rollout.weaknesses:
                category_counts[weakness] += 1
                if len(category_examples[weakness]) < 5:
                    category_examples[weakness].append(rollout.problem[:200])

        diagnoses = []
        for category, count in category_counts.most_common():
            if count < min_count:
                continue
            diagnoses.append(WeaknessDiagnosis(
                category=category,
                description=f"Model struggles with {category.replace('_', ' ')} "
                            f"({count} failures in {len(incorrect)} incorrect rollouts)",
                count=count,
                example_problems=category_examples[category],
            ))

        return diagnoses

    # ------------------------------------------------------------------
    # Step 3: Build embedding index from training problems
    # ------------------------------------------------------------------

    def build_problem_index(
        self,
        dataset_names: list[str],
        cache_dir: Optional[str] = None,
    ) -> int:
        """Create sentence embeddings for all problems using sentence-transformers.

        Loads problems from HuggingFace datasets (MATH, gsm8k, etc.),
        encodes them, and builds a FAISS index for fast similarity search.

        Args:
            dataset_names: List of dataset names to load.
                Supported: "MATH", "gsm8k", "AIME" (custom parquet).
            cache_dir: Optional cache directory for embeddings.

        Returns:
            Number of problems indexed.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for the embedding index. "
                "Install with: pip install faiss-cpu  (or faiss-gpu)"
            )

        from datasets import load_dataset

        self._problems = []

        for name in dataset_names:
            if name.upper() == "MATH":
                ds = load_dataset("hendrycks/competition_math", split="train")
                for row in tqdm(ds, desc="Loading MATH"):
                    self._problems.append(ProblemEntry(
                        text=row["problem"],
                        source="MATH",
                        subject=row.get("type", "unknown"),
                        difficulty=str(row.get("level", "unknown")),
                        solution=row.get("solution"),
                    ))
            elif name.lower() == "gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split="train")
                for row in tqdm(ds, desc="Loading GSM8K"):
                    self._problems.append(ProblemEntry(
                        text=row["question"],
                        source="gsm8k",
                        subject="arithmetic",
                        difficulty="easy",
                        solution=row.get("answer"),
                    ))
            elif Path(name).exists():
                # Custom parquet file
                df = pd.read_parquet(name)
                for _, row in tqdm(df.iterrows(), desc=f"Loading {name}", total=len(df)):
                    self._problems.append(ProblemEntry(
                        text=row.get("question", row.get("problem", "")),
                        source=Path(name).stem,
                        subject=row.get("subject", "unknown"),
                        difficulty=row.get("difficulty", "unknown"),
                        solution=row.get("solution", row.get("answer", None)),
                    ))
            else:
                print(f"Warning: Unknown dataset '{name}', skipping.")

        if not self._problems:
            raise ValueError("No problems loaded. Check dataset names.")

        # Encode all problems
        print(f"Encoding {len(self._problems)} problems...")
        texts = [p.text for p in self._problems]

        # Use cache if available
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"embeddings_{self._embedding_model_name}_{len(texts)}.npy"
            if cache_path.exists():
                print(f"Loading cached embeddings from {cache_path}")
                embeddings = np.load(str(cache_path))
                if len(embeddings) == len(texts):
                    self._build_faiss_index(embeddings)
                    return len(self._problems)

        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=256,
            normalize_embeddings=True,
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embeddings)
            print(f"Cached embeddings to {cache_path}")

        self._build_faiss_index(embeddings)
        return len(self._problems)

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings."""
        import faiss

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized vecs)
        self._index.add(embeddings.astype(np.float32))
        print(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------
    # Step 4: Search similar problems for each weakness
    # ------------------------------------------------------------------

    def search_similar(
        self,
        weakness: WeaknessDiagnosis,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Find similar problems by embedding similarity.

        Uses the weakness category description + example problems as the
        query to find training problems that exercise the same skills.

        Args:
            weakness: A WeaknessDiagnosis from extract_weaknesses().
            top_k: Number of similar problems to return.

        Returns:
            List of dicts with keys: text, source, subject, difficulty, score.
        """
        if self._index is None:
            raise RuntimeError("Call build_problem_index() before search_similar()")

        # Build query from weakness description + example problems
        query_parts = [
            f"Math problem involving {weakness.category.replace('_', ' ')}",
        ]
        for ex in weakness.example_problems[:3]:
            query_parts.append(ex)
        query_text = " ".join(query_parts)

        query_vec = self.encoder.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            prob = self._problems[idx]
            results.append({
                "text": prob.text,
                "source": prob.source,
                "subject": prob.subject,
                "difficulty": prob.difficulty,
                "solution": prob.solution,
                "similarity_score": float(score),
            })

        return results

    # ------------------------------------------------------------------
    # Step 5: Build targeted curriculum
    # ------------------------------------------------------------------

    def build_curriculum(
        self,
        weaknesses: list[WeaknessDiagnosis],
        top_k: int = 50,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        """Combine all weakness-targeted problems into a training dataset.

        For each diagnosed weakness, retrieves the top_k most similar
        training problems and merges them into a single curriculum dataset.

        Args:
            weaknesses: List of WeaknessDiagnosis from extract_weaknesses().
            top_k: Number of problems to retrieve per weakness.
            deduplicate: Whether to remove duplicate problems across categories.

        Returns:
            DataFrame with columns: text, source, subject, difficulty,
            weakness_category, similarity_score, solution.
        """
        all_problems = []

        for weakness in weaknesses:
            print(f"Searching for {weakness.category} "
                  f"(count={weakness.count})...")
            results = self.search_similar(weakness, top_k=top_k)
            for r in results:
                r["weakness_category"] = weakness.category
                r["weakness_count"] = weakness.count
            all_problems.extend(results)

        df = pd.DataFrame(all_problems)

        if deduplicate and len(df) > 0:
            before = len(df)
            # Keep the entry with highest similarity score for each problem
            df = df.sort_values("similarity_score", ascending=False)
            df = df.drop_duplicates(subset=["text"], keep="first")
            print(f"Deduplicated: {before} -> {len(df)} problems")

        # Sort by weakness importance (count) then similarity
        if len(df) > 0:
            df = df.sort_values(
                ["weakness_count", "similarity_score"],
                ascending=[False, False],
            )

        print(f"\n=== Curriculum Summary ===")
        print(f"Total problems: {len(df)}")
        if len(df) > 0:
            print(f"By weakness category:")
            for cat, group in df.groupby("weakness_category"):
                print(f"  {cat}: {len(group)} problems")
            print(f"By source:")
            for src, group in df.groupby("source"):
                print(f"  {src}: {len(group)} problems")

        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_answer(response: str, expected: str) -> bool:
    """Simple answer check: extract \\boxed{} and compare."""
    if not expected:
        return False

    # Extract boxed answer
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if not boxed_match:
        return False

    predicted = boxed_match.group(1).strip()
    expected = expected.strip()

    # Normalize: remove spaces, compare
    return predicted.replace(" ", "") == expected.replace(" ", "")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI for building a curriculum from pre-computed rollouts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build targeted curriculum from weakness diagnoses"
    )
    parser.add_argument(
        "--rollouts", type=str, required=True,
        help="Path to rollouts parquet (columns: question, response, correct, meta_blocks)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["MATH", "gsm8k"],
        help="Datasets to build index from (default: MATH gsm8k)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Number of similar problems per weakness category",
    )
    parser.add_argument(
        "--min-count", type=int, default=3,
        help="Minimum failure count to consider a weakness real",
    )
    parser.add_argument(
        "--output", type=str, default="sft_data/curriculum.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory to cache embeddings",
    )
    args = parser.parse_args()

    # Load pre-computed rollouts
    df_rollouts = pd.read_parquet(args.rollouts)
    rollouts = []
    for _, row in df_rollouts.iterrows():
        meta_blocks_raw = row.get("meta_blocks", "[]")
        if isinstance(meta_blocks_raw, str):
            meta_blocks = json.loads(meta_blocks_raw)
        else:
            meta_blocks = list(meta_blocks_raw) if meta_blocks_raw is not None else []

        weaknesses = []
        for block in meta_blocks:
            weaknesses.extend(classify_weakness(block))

        rollouts.append(Rollout(
            problem=row.get("question", row.get("problem", "")),
            response=row.get("response", ""),
            correct=bool(row.get("correct", False)),
            meta_blocks=meta_blocks,
            weaknesses=list(set(weaknesses)),
        ))

    # Build curriculum
    rag = CurriculumRAG(embedding_model=args.embedding_model)

    print(f"\n=== Step 1: Extract weaknesses from {len(rollouts)} rollouts ===")
    weaknesses = rag.extract_weaknesses(rollouts, min_count=args.min_count)
    for w in weaknesses:
        print(f"  {w.category}: {w.count} failures")

    if not weaknesses:
        print("No significant weaknesses found. Exiting.")
        return

    print(f"\n=== Step 2: Build problem index from {args.datasets} ===")
    n_indexed = rag.build_problem_index(args.datasets, cache_dir=args.cache_dir)
    print(f"Indexed {n_indexed} problems")

    print(f"\n=== Step 3: Build curriculum ===")
    curriculum = rag.build_curriculum(weaknesses, top_k=args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum.to_parquet(output_path, index=False)
    print(f"\nSaved curriculum to {output_path}")


if __name__ == "__main__":
    main()
