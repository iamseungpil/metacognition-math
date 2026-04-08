# Math Benchmark Survey for LLM Evaluation (Beyond GSM8K / MATH-500)

Survey date: 2026-03-31

## Summary Table

| Benchmark | HuggingFace ID | # Problems | Difficulty | Format |
|-----------|---------------|------------|------------|--------|
| MATH (full) | `hendrycks/competition_math` | 12,500 | Competition (AMC 10/12, AIME) | Open-ended, 7 subjects, levels 1-5 |
| AIME 2024 | `math-ai/aime24` | 30 | Olympiad-qualifying | Integer 0-999 |
| AIME 2025 | `math-ai/aime25`, `MathArena/aime_2025` | 30 | Olympiad-qualifying | Integer 0-999 |
| OlympiadBench | `Hothan/OlympiadBench` | 8,952 | International Olympiad | Open-ended + proofs, bilingual (EN/CN) |
| Omni-MATH | `KbsdJames/Omni-MATH` | 4,428 | Olympiad (33 sub-domains, 10+ difficulty levels) | Open-ended |
| MathOdyssey | (GitHub/paper release) | 387 | High-school to Olympiad | Open-ended with solutions |
| MGSM | `juletxara/mgsm` | 250 x 10 langs | Grade-school (= GSM8K translated) | Multilingual, numeric answer |
| MMLU-Pro (math portion) | `TIGER-Lab/MMLU-Pro` | ~5,122 math items (of 12,032 total) | Undergraduate | 10-choice MCQ |
| LiveMathBench | `opencompass/LiveMathBench` | 238 (202412 version) | Competition (AMC, CNMO, Putnam) | Open-ended, rolling updates |
| MathArena | `MathArena/*` (aime_2025, hmmt_feb_2025, etc.) | 162+ across 7 competitions | Competition to Olympiad | Open-ended, uncontaminated |
| GPQA Diamond | `idavidrein/gpqa` | 198 (Diamond subset) | Graduate-level | 4-choice MCQ (physics/chem/bio, not math) |

## Detailed Notes

### Tier 1: Recommended for Hard Math Evaluation

**1. Omni-MATH** -- Best option for Olympiad-level math
- HF: `KbsdJames/Omni-MATH`
- 4,428 problems, 33+ sub-domains, 10+ difficulty levels
- o1-mini scores 60.5%, o1-preview scores 52.6% -- genuinely hard
- Source: https://huggingface.co/datasets/KbsdJames/Omni-MATH

**2. OlympiadBench** -- Largest Olympiad-level collection
- HF: `Hothan/OlympiadBench`
- 8,952 problems (math + physics), bilingual EN/CN
- GPT-4V scores 17.97% overall -- extremely challenging
- Includes proofs and open-ended questions
- Source: https://huggingface.co/datasets/Hothan/OlympiadBench

**3. AIME 2025 (via MathArena)** -- Uncontaminated competition math
- HF: `MathArena/aime_2025` or `math-ai/aime25`
- 30 problems per year, integer answers 0-999
- AIME 2024 is contaminated in many LLMs; AIME 2025 is cleaner
- MathArena provides rolling uncontaminated evaluations
- Source: https://huggingface.co/datasets/MathArena/aime_2025

**4. MATH (full, 12,500)** -- Standard hard math benchmark
- HF: `hendrycks/competition_math`
- 12,500 problems from AMC 10/12, AIME, etc.
- 7 subjects: Algebra, Number Theory, Geometry, Counting & Probability, Precalculus, Intermediate Algebra, Prealgebra
- Difficulty levels 1-5; level 4-5 problems remain challenging
- Source: https://huggingface.co/datasets/hendrycks/competition_math

### Tier 2: Useful Complements

**5. LiveMathBench** -- Anti-contamination rolling benchmark
- HF: `opencompass/LiveMathBench`
- 238 problems (202412 version) from CNMO, AMC, Putnam
- Regularly updated with fresh competition problems
- Source: https://huggingface.co/datasets/opencompass/LiveMathBench

**6. MathArena** -- Multi-competition evaluation suite
- HF: `MathArena/*` (multiple competition datasets)
- 162+ problems across 7 competitions (AIME, HMMT, APEX, IMO)
- Top models achieve ~40% on IMO 2025
- Source: https://matharena.ai/

**7. MathOdyssey** -- Expert-curated diverse difficulty
- 387 problems spanning high-school to Olympiad
- Each problem has detailed solution and categorization
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12334620/

**8. MMLU-Pro (math subset)** -- Undergraduate-level MCQ
- HF: `TIGER-Lab/MMLU-Pro`
- ~5,122 math items requiring multi-step calculation
- 10-option MCQ format (harder than 4-option MMLU)
- Source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

### Tier 3: Complementary / Different Purpose

**9. MGSM** -- Multilingual (not harder, but different axis)
- HF: `juletxara/mgsm`
- 250 GSM8K problems translated into 10 languages
- Tests multilingual math reasoning, not harder math per se
- Source: https://huggingface.co/datasets/juletxara/mgsm

**10. GPQA Diamond** -- Graduate science (not pure math)
- HF: `idavidrein/gpqa` (gated)
- 198 expert-written questions in physics, chemistry, biology
- Experts score 74%, non-experts score 34%
- Not math-specific but tests quantitative reasoning
- Source: https://huggingface.co/datasets/idavidrein/gpqa

## Recommendation for Meta-CoT Evaluation

For evaluating Meta-CoT on harder math beyond GSM8K and MATH-500:
1. **Primary**: Omni-MATH (4,428 problems, true Olympiad difficulty, well-structured)
2. **Secondary**: AIME 2025 via MathArena (30 problems, uncontaminated, integer answers -- easy to evaluate)
3. **Scale test**: MATH full dataset levels 4-5 only (filter from `hendrycks/competition_math`)
4. **Ceiling test**: OlympiadBench math subset (if models score well on Omni-MATH)
