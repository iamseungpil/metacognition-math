# Singularity `msrresrchbasicvc` rejects all new job submissions — escalation report

**Filed:** 2026-07-26 · **Symptom start:** ~05:49 UTC 2026-07-26 · **Still failing at:** 18:04 UTC (14 consecutive probes)

---

## One-line summary

Every new job submission to **`msrresrchbasicvc`** is rejected at the
`managementfrontend` with `UserError: The virtual cluster does not exist`,
**including a 1-CPU job whose entire command is two `echo` lines**. Jobs already
admitted to that VC keep running normally. Submissions to `msrresrchvc` from the
same account, client, workspace, and project succeed immediately.

---

## Reproduction (minimal)

`amlt run` with this config fails every time:

```yaml
target:
  service: sing
  name: msrresrchbasicvc
  workspace_name: msra-sh-aml-ws
environment:
  registry: mcr.microsoft.com
  image: aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:latest
jobs:
  - name: canary
    sku: C1
    sla_tier: Basic
    priority: high
    identity: managed
    submit_args:
      max_run_duration_seconds: 300
    command:
      - echo "BASICVC RECOVERED"
      - date -u
```

Changing **only** `name: msrresrchbasicvc` → `msrresrchvc` (and the SKU to one
that VC offers) makes the identical submission succeed and reach `preparing`
within one second.

## Error response

```
HttpResponseError: (UserError) The virtual cluster does not exist. Please ensure
that you submitted a job to a virtual cluster which exists in the same cloud.
A virtualcluster created in one Cloud cannot be used to submit jobs to another cloud.
Code: UserError
ComponentName: managementfrontend
Environment: westus2
Location:    westus2
```

Correlation IDs (each from a separate attempt):

| UTC time | operation | request |
|---|---|---|
| 2026-07-26T18:04:00 | `cebe26f9d9105d31e359ad11ce058a18` | `478560a673cba078` |
| 2026-07-26T10:40:16 | `b0a9778f0c3a7bb3f291fc4c85dfcf6a` | `e82a2071d043a1a5` |
| 2026-07-26T10:38:17 | `d314cc7f11b8828d0a41af0e55f0f736` | `57264fbfa04b30b0` |
| 2026-07-26T10:36:57 | `de4f84a1a8eb57ba6acdeb7140c285be` | `02b58f89ef18115e` |
| 2026-07-26T10:35:32 | `a833fdc4efd30f9e5c7e3b82264ea66e` | `96de55b508c56afd` |

## Environment

| | |
|---|---|
| VC | `msrresrchbasicvc` — subscription **Singularity Shared** (`22da88f6-1210-4de2-a5a3-da4c7c2a1213`), RG `gcr-singularity` |
| Workspace | `msra-sh-aml-ws` — subscription `c4c534bc-9978-4974-9c87-551f7c5754ef`, RG `msra-sh-aml-rg` |
| Identity | managed, UAI `msra-sh-aml-uai` |
| Client | Amulet **11.9.1** and **11.16.0**, both fail identically |
| User | `sc-vhr286860@microsoft.com` |

## What we ruled out (each verified, not assumed)

| Hypothesis | How it was tested | Result |
|---|---|---|
| Bad job config / code / data | Resubmitted the exact YAML that succeeded on this VC 14 h earlier | ❌ same error |
| Wrong SLA tier | Submitted `sla_tier: Standard` as well as `Basic` | ❌ both fail |
| Stale amlt client | Installed **11.16.0** in a clean venv, resubmitted | ❌ same error |
| Region / cloud mismatch (what the message claims) | `az resource show` on both objects | ❌ **VC and workspace are both `westus2`** — the message is not describing the real cause |
| GPU capacity or quota | Submitted a **1-CPU** job (`sku: C1`) running only `echo` | ❌ same error — the rejection is not GPU-related |
| Account / token / workspace / project storage | Same account + client + workspace + project submitted to `msrresrchvc` | ✅ **succeeds**, reaches `preparing` in ~1 s |
| VC deleted or unreachable | `amlt target list sing` | ✅ enumerates `msrresrchbasicvc` with A100, CPU, H100, H200, MI200, MI300X, V100 |
| Whole account cut off | Our job `rq3v2_b2p` admitted before the change | ✅ **still running on `msrresrchbasicvc` after 24 h** |

**Conclusion:** the failure is specific to *new admissions* to `msrresrchbasicvc`
for this identity. Existing admissions are unaffected, and the error text about
clouds/regions is not consistent with the measured region of either object.

## Impact

Our H100/H200 work runs on this VC. The alternative we can reach
(`msrresrchvc`) offers A100/CPU/MI200 only, and all of its A100 SKUs are `Basic`
tier — three of our jobs have been queued there for 3–4 h without being
scheduled. The affected work is a three-arm training ladder estimated at roughly
six days per arm on A100.

## Ask

1. Whether entitlement/RBAC for `msrresrchbasicvc` changed for this identity or
   for the `Singularity Shared` subscription around **05:49 UTC 2026-07-26**.
2. If the block is intentional, what the correct target or tier is now.
3. If it is not intentional, restoration of submission access.

---

*Evidence in this report is reproducible with the YAML above. Fuller narrative,
including the falsification sequence, is in
`docs/reports/2026-07-17-rq3-run-and-iteration-log.md` entries E-091 and E-099.*
