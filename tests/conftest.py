"""Eagerly import heavy real deps BEFORE any test-level sys.modules stubbing.

Several test modules (e.g. test_dcpo_v3_cf's _AutoStubFinder, the verl/ray
stub helpers) mutate sys.modules / sys.meta_path at import time. transformers
is a lazy package, so if the first FULL resolution of transformers.generation
or torch.distributed.tensor happens after those stubs are installed, collection
fails with "kernel already registered for wait_tensor" / "cannot import
GenerationMixin". Historically tests/test_contrastive_meta_rlsd.py (archived
2026-07-12 to archive/dead_code_2026_07_12/) shadowed this by importing
trl->transformers first; this conftest makes that preload explicit.
"""
import torch.distributed.tensor  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
