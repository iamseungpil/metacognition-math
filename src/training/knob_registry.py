"""Launch-time validation of the dcpo_* knob surface against core/KNOBS.yaml.

WHY THIS EXISTS
---------------
2026-06-22 commit f1f6cec set ``dcpo_meta_floor`` 0.05 -> 0.0 as a side effect of
cf_group work. Six weeks later a pmi_shift run inherited that value, meta emission
fell 1.00 -> 0.018, and the second half of a 300-step run measured a treatment that
was no longer there. Nothing failed loudly because:

  * the code default is ``config.get("dcpo_meta_floor", 0.0)`` — deleting the key and
    disabling the guard are the SAME observable state, and
  * the value lives in an inherited config, so a launcher-level matched audit that
    diffs ``++algorithm.*`` overrides byte for byte cannot see it.

This module makes both of those impossible to repeat. It does not try to decide
whether a value is *correct* — only that no load-bearing knob is invisible and no
retired lineage is silently reachable.

CONTRACT
--------
``validate(algorithm_cfg)`` raises ``KnobRegistryError`` when:

  1. a ``dead_lineage`` knob deviates from its registered default
     (that is the only way to reach retired reward generations),
  2. ``dcpo_ack_load_bearing`` does not name exactly the load-bearing set
     (forces whoever launches to type the names, so the values get printed), or
  3. ``dcpo_rmeta_source`` is absent or outside the registry's legal values.

On success it returns the resolved value of every live knob so the caller can log
them; provenance then lives in the run log instead of in nobody's head.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import yaml

_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "core", "KNOBS.yaml",
)

# Sections of core/KNOBS.yaml that carry per-knob entries.
_ENTRY_SECTIONS = ("load_bearing", "live", "default_only", "dead_lineage")

_ACK_KEY = "dcpo_ack_load_bearing"


class KnobRegistryError(RuntimeError):
    """A knob-surface violation that must stop the launch."""


def _load(path: str | None = None) -> Dict[str, Any]:
    with open(path or _REGISTRY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _entries(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten the registry to {knob_name: entry} for entries that name a status."""
    out: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            if isinstance(val, dict) and "status" in val:
                out[key] = val
            else:
                walk(val)

    for section in _ENTRY_SECTIONS:
        walk(registry.get(section))
    return out


def _read(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(name, default)
        except Exception:
            pass
    return getattr(cfg, name, default)


def _present(cfg: Any, name: str) -> bool:
    sentinel = object()
    return _read(cfg, name, sentinel) is not sentinel


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b or (a is None and b is None)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def validate(algorithm_cfg: Any, registry_path: str | None = None) -> List[Tuple[str, Any]]:
    """Validate the knob surface. Returns [(live_knob, resolved_value)] on success."""
    registry = _load(registry_path)
    entries = _entries(registry)
    if not entries:
        raise KnobRegistryError(f"knob registry at {registry_path or _REGISTRY_PATH} has no entries")

    problems: List[str] = []

    # (1) retired lineages must stay at their registered default.
    for name, entry in entries.items():
        if not str(entry.get("status", "")).startswith("dead"):
            continue
        if not _present(algorithm_cfg, name):
            continue
        value = _read(algorithm_cfg, name)
        if not _same(value, entry.get("default")):
            problems.append(
                f"{name}={value!r} deviates from its registered default "
                f"{entry.get('default')!r}; that knob belongs to a retired reward "
                f"generation and setting it reaches dead code"
            )

    # (2) every load-bearing knob must be named in the acknowledgement list, so the
    #     launcher cannot inherit one invisibly.
    load_bearing = {
        name for name, entry in entries.items()
        if name in (registry.get("load_bearing") or {})
    }
    if load_bearing:
        acked = _read(algorithm_cfg, _ACK_KEY, None)
        acked_set = set(acked) if isinstance(acked, (list, tuple)) else set()
        missing = sorted(load_bearing - acked_set)
        extra = sorted(acked_set - load_bearing)
        if missing:
            problems.append(
                f"{_ACK_KEY} does not name load-bearing knob(s) {missing}; add them to the "
                f"launcher so their resolved values are printed at launch"
            )
        if extra:
            problems.append(f"{_ACK_KEY} names unknown knob(s) {extra}")

    # (3) the source selector must be present and legal.
    src_entry = entries.get("dcpo_rmeta_source")
    if src_entry is not None:
        if not _present(algorithm_cfg, "dcpo_rmeta_source"):
            problems.append(
                "dcpo_rmeta_source is absent; it selects which of the mutually exclusive "
                "meta-reward generations runs and has no safe default"
            )
        else:
            legal_raw = str(src_entry.get("legal_values", ""))
            legal = {tok.strip() for tok in legal_raw.replace("(", " ").split()[:0] or []}
            if not legal:
                legal = {
                    tok.strip().rstrip(",")
                    for tok in legal_raw.split()
                    if tok.strip().rstrip(",").isidentifier()
                }
            value = str(_read(algorithm_cfg, "dcpo_rmeta_source"))
            if legal and value not in legal:
                problems.append(f"dcpo_rmeta_source={value!r} not in registry legal values {sorted(legal)}")

    if problems:
        raise KnobRegistryError(
            "knob registry validation failed (core/KNOBS.yaml):\n  - " + "\n  - ".join(problems)
        )

    resolved: List[Tuple[str, Any]] = []
    for name, entry in sorted(entries.items()):
        if str(entry.get("status", "")) == "live":
            resolved.append((name, _read(algorithm_cfg, name, entry.get("default"))))
    return resolved
