#!/usr/bin/env python3
"""Resolve the current SSH ProxyEndpoint for an AML job display name."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _run_json(cmd: list[str]) -> object:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-name", required=True, help="AML display_name, e.g. metacognition_eval")
    parser.add_argument("--workspace-name", default="msra-sh-aml-ws")
    parser.add_argument("--resource-group", default="msra-sh-aml-rg")
    args = parser.parse_args()

    rows = _run_json([
        "az", "ml", "job", "list",
        "--workspace-name", args.workspace_name,
        "--resource-group", args.resource_group,
        "--max-results", "200",
        "-o", "json",
    ])
    if not isinstance(rows, list):
        raise RuntimeError("az ml job list did not return a list")

    matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("display_name") == args.display_name
        and row.get("status") in {"Running", "Queued", "Preparing"}
    ]
    if not matches:
        raise RuntimeError(f"No active AML job found for display_name={args.display_name!r}")

    def _created_at(row: dict) -> str:
        ctx = row.get("creation_context") or {}
        return str(ctx.get("created_at") or "")

    job = sorted(matches, key=_created_at, reverse=True)[0]
    services = _run_json([
        "az", "ml", "job", "show-services",
        "--workspace-name", args.workspace_name,
        "--resource-group", args.resource_group,
        "--name", job["name"],
        "-o", "json",
    ])
    proxy = (((services or {}).get("SSH") or {}).get("properties") or {}).get("ProxyEndpoint")
    if not proxy:
        raise RuntimeError(f"Active job {job['name']} has no SSH ProxyEndpoint")
    sys.stdout.write(str(proxy))


if __name__ == "__main__":
    main()
