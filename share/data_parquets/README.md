# 공유용 코퍼스 (2026-08-03)

`data/`는 `.gitignore`의 `/data/` 규칙으로 추적되지 않는다. 협업자가 바로 받을 수 있도록
현행 사다리가 쓰는 parquet 8종을 여기 복사해 커밋한다. **정본은 HuggingFace**이며 이 폴더는 편의 사본이다.

| 파일 | 행 | 쓰이는 곳 | HF 정본 |
|---|---|---|---|
| `v8_meta_inside_strict.parquet` | 4,264 | **instruct SFT1 (메타)** — 성공한 계보의 1단 | `metacot`(dataset) |
| `v8_base_matched_strict.parquet` | 4,264 | instruct SFT1 (무메타 쌍둥이) | `metacot`(dataset) |
| `b2on_v8meta_strict_sft.parquet` | 4,245 | **base SFT1 (메타)** | `metacot-rv`(dataset) |
| `b0on_v8base_strict_sft.parquet` | 4,245 | base SFT1 (무메타 쌍둥이) | `metacot-rv`(dataset) |
| `rv_redirect_verify_functional.parquet` | **1,763** | **SFT2 (메타) — instruct·base 공통** | `metacot-rv`(dataset) |
| `v8_base_rv_sft.parquet` | **1,763** | SFT2 (무메타 쌍둥이) — 위와 행 단위 대응 | `metacot`(dataset) |
| `verl_train_meta_mix.parquet` | 5,344 | RL 학습 | `metacot-sdc-data`(dataset) |
| `verl_val_meta_mix.parquet` | 594 | RL in-training val | `metacot-sdc-data`(dataset) |

## 읽는 법

- **SFT1 두 개는 hard 1,340행을 포함**한다. **SFT2 두 개는 easy 870 + medium 893, hard 0건**이다.
  그래서 이 프로젝트의 "분포 밖" 축은 데이터셋 정체가 아니라 **난이도**이고,
  MATH500 level 4–5(262문항)가 그 축의 표본이다.
- SFT2 두 파일은 **행 단위로 대응**한다(둘 다 1,763행, verify 1,209 / redirect 554,
  easy 870 / medium 893). 차이는 메타 블록의 유무뿐이다 — `configs/archive/sft_base_rv.yaml`의
  주석이 *"differs ONLY by the absence of meta"*라고 명시한다.
- `verl_val_meta_mix`는 **판정에 쓰지 않는다**. 벤치별 셀이 21~38문항이라 한 문제가 2.6~4.8pp다.
