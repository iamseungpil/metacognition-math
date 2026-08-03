# Paired held-out eval — T1 (accuracy) + T3-style tables

- eval-dir: /tmp/claude-587327809/-home-v-seungplee/cbc3d41a-829b-4e70-aaba-08908a143cb2/scratchpad/eval_flat
- grading: math_verify robust regrade; accuracy = avg@k (per-problem mean over samples, then macro over problems)
- AIME @16k = union of seed42 (pass b) + seed43 (pass c) = avg@16 by question (no double count)
- meta emission counts CLOSED <|meta|>...<|/meta|> blocks only; truncation = finish_reason == "length"

## Coverage (what was loaded)

| file | arm | budget | seed | benchmark | rows |
|---|---|---|---|---|---|
| pmishift_gs300_16k_n8_aime2024.parquet | pmishift | 16k | 42 | aime2024 | 240 |
| pmishift_gs300_16k_n8_seed43_aime.parquet | pmishift | 16k | 43 | aime2024 | 240 |
| pmishift_gs300_4k_n8.parquet | pmishift | 4k | 42 | (all) | 8240 |

## T1 — main accuracy (robust avg@k; rtΔ = runtime − robust)

| benchmark | budget | k | pmishift avg@k | pmishift rtΔ |
|---|---|---|---|---|
| gsm8k | 4k | 8 | 93.9% | -1.4pp |
| math500 | 4k | 8 | 81.5% | -15.4pp |
| aime2024 | 4k | 8 | 19.2% | +0.0pp |
| aime2024 | 16k | 16 | 18.5% | +0.0pp |

## Per-cell details (truncation, tokens, meta emission)

| arm | benchmark | budget | k | n prob | avg@k (robust) | avg@k (runtime) | trunc% | mean tokens | meta% |
|---|---|---|---|---|---|---|---|---|---|
| pmishift | gsm8k | 4k | 8 | 500 | 93.9% | 92.5% | 0.0% | 340 | 87.8% |
| pmishift | math500 | 4k | 8 | 500 | 81.5% | 66.0% | 6.8% | 1061 | 90.7% |
| pmishift | aime2024 | 4k | 8 | 30 | 19.2% | 19.2% | 48.8% | 3283 | 95.8% |
| pmishift | aime2024 | 16k | 16 | 30 | 18.5% | 18.5% | 50.0% | 9504 | 98.1% |

## Paired significance (pmishift − base, shared problems)

_Paired significance needs BOTH arms for a benchmark x budget; none present._
