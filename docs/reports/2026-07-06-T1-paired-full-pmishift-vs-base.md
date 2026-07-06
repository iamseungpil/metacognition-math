# Paired held-out eval — T1 (accuracy) + T3-style tables

- eval-dir: /tmp/claude-587327809/-home-v-seungplee/cbc3d41a-829b-4e70-aaba-08908a143cb2/scratchpad/eval_flat
- grading: math_verify robust regrade; accuracy = avg@k (per-problem mean over samples, then macro over problems)
- AIME @16k = union of seed42 (pass b) + seed43 (pass c) = avg@16 by question (no double count)
- meta emission counts CLOSED <|meta|>...<|/meta|> blocks only; truncation = finish_reason == "length"

## Coverage (what was loaded)

| file | arm | budget | seed | benchmark | rows |
|---|---|---|---|---|---|
| base_gs300_16k_n8_aime2024.parquet | base | 16k | 42 | aime2024 | 240 |
| base_gs300_16k_n8_gsm8k.parquet | base | 16k | 42 | gsm8k | 4000 |
| base_gs300_16k_n8_math500.parquet | base | 16k | 42 | math500 | 4000 |
| base_gs300_16k_n8_seed43_aime.parquet | base | 16k | 43 | aime2024 | 240 |
| base_gs300_4k_n8.parquet | base | 4k | 42 | (all) | 8240 |
| pmishift_gs300_16k_n8_aime2024.parquet | pmishift | 16k | 42 | aime2024 | 240 |
| pmishift_gs300_16k_n8_gsm8k.parquet | pmishift | 16k | 42 | gsm8k | 4000 |
| pmishift_gs300_16k_n8_math500.parquet | pmishift | 16k | 42 | math500 | 4000 |
| pmishift_gs300_16k_n8_seed43_aime.parquet | pmishift | 16k | 43 | aime2024 | 240 |
| pmishift_gs300_4k_n8.parquet | pmishift | 4k | 42 | (all) | 8240 |

## T1 — main accuracy (robust avg@k; rtΔ = runtime − robust)

| benchmark | budget | k | pmishift avg@k | pmishift rtΔ | base avg@k | base rtΔ |
|---|---|---|---|---|---|---|
| gsm8k | 4k | 8 | 93.9% | -1.4pp | 89.9% | -1.2pp |
| gsm8k | 16k | 8 | 93.3% | -1.4pp | 90.2% | -1.2pp |
| math500 | 4k | 8 | 81.5% | -15.4pp | 62.7% | -8.9pp |
| math500 | 16k | 8 | 81.8% | -15.8pp | 62.9% | -8.5pp |
| aime2024 | 4k | 8 | 19.2% | +0.0pp | 5.0% | +0.0pp |
| aime2024 | 16k | 16 | 18.5% | +0.0pp | 4.8% | +0.0pp |

## Per-cell details (truncation, tokens, meta emission)

| arm | benchmark | budget | k | n prob | avg@k (robust) | avg@k (runtime) | trunc% | mean tokens | meta% |
|---|---|---|---|---|---|---|---|---|---|
| pmishift | gsm8k | 4k | 8 | 500 | 93.9% | 92.5% | 0.0% | 340 | 87.8% |
| pmishift | gsm8k | 16k | 8 | 500 | 93.3% | 92.0% | 0.0% | 341 | 87.9% |
| pmishift | math500 | 4k | 8 | 500 | 81.5% | 66.0% | 6.8% | 1061 | 90.7% |
| pmishift | math500 | 16k | 8 | 500 | 81.8% | 66.0% | 6.0% | 1805 | 90.4% |
| pmishift | aime2024 | 4k | 8 | 30 | 19.2% | 19.2% | 48.8% | 3283 | 95.8% |
| pmishift | aime2024 | 16k | 16 | 30 | 18.5% | 18.5% | 50.0% | 9504 | 98.1% |
| base | gsm8k | 4k | 8 | 500 | 89.9% | 88.7% | 0.4% | 183 | 0.0% |
| base | gsm8k | 16k | 8 | 500 | 90.2% | 89.0% | 0.4% | 229 | 0.0% |
| base | math500 | 4k | 8 | 500 | 62.7% | 53.8% | 19.0% | 1050 | 0.0% |
| base | math500 | 16k | 8 | 500 | 62.9% | 54.4% | 18.8% | 3367 | 0.0% |
| base | aime2024 | 4k | 8 | 30 | 5.0% | 5.0% | 74.2% | 3272 | 0.0% |
| base | aime2024 | 16k | 16 | 30 | 4.8% | 4.8% | 72.9% | 12193 | 0.0% |

## Paired significance (pmishift − base, shared problems)

| benchmark | budget | n shared | effect (A-B) | 95% CI | boot p | McNemar b/c | McNemar p |
|---|---|---|---|---|---|---|---|
| gsm8k | 4k | 500 | +4.0pp | [+2.5, +5.6]pp | 0.000 | 27/6 | 0.000 |
| gsm8k | 16k | 500 | +3.1pp | [+1.6, +4.7]pp | 0.000 | 21/8 | 0.024 |
| math500 | 4k | 500 | +18.8pp | [+16.2, +21.4]pp | 0.000 | 100/4 | 0.000 |
| math500 | 16k | 500 | +18.9pp | [+16.3, +21.6]pp | 0.000 | 100/2 | 0.000 |
| aime2024 | 4k | 30 | +14.2pp | [+5.8, +23.8]pp | 0.000 | 6/0 | 0.031 |
| aime2024 | 16k | 30 | +13.8pp | [+5.8, +23.1]pp | 0.000 | 5/0 | 0.062 |
