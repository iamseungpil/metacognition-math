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

---

## UPDATE 2026-07-27 — the block is a GROUP POLICY membership gap, not a missing VC

A user in the SAME workspace (`msra-sh-aml-ws`), submitting to the SAME virtual
cluster (`msrresrchbasicvc`), is running an H100 job right now:

```
displayName: debug            status: Running       StartTimeUtc: 2026-07-25 22:05:02
tags:  GroupPolicy: e9deff52-56e7-4074-bb58-056bbd931bb6
properties:  amlt.job.sku: G4-H100
             azureml.SLATier: Standard
             azureml.InstanceType: Singularity.ND48_H100_v5
             azureml.VC: .../virtualclusters/msrresrchbasicvc
```

Their job carries a **`GroupPolicy` tag**. Ours do not. Amulet exposes this as
`-t NAME[:GROUP_POLICY]`, and resolving that policy for our identity fails:

```
$ amlt cache expand-sku -t "msrresrchbasicvc:e9deff52-56e7-4074-bb58-056bbd931bb6" 80G4-H100 --sla Standard
"msrresrchbasicvc:e9deff52-56e7-4074-bb58-056bbd931bb6" could not be found.
```

### What this changes about the diagnosis

Everything previously reported still holds — the VC resource is readable
(`az resource show` returns `westus2`), the SKU resolves
(`80G4-H100 -> Singularity.ND48_H100_v5` under both Standard and Basic), and
`amlt ti sing` still shows H100 quota against our identity (Standard 307/560,
Basic 29/1120, user max 16). What is new is *why* those can all be true while
submission fails: the quota we can see is issued by the VC's **default** group
policy, and that default policy no longer admits us for job submission. The
"virtual cluster does not exist" text is how `managementfrontend` reports a
policy/entitlement miss, which is why it appeared identically for a 1-CPU echo
job and for H100.

Two further observations that were not in the original report:

- `msrresrchbasicvc` has **no Premium SLA on any instance**. Probing Premium
  returns `Selected target 'msrresrchbasicvc' has no Premium SLA for any
  instance` and lists exactly one resolvable SKU, `3C1` (Dv3 CPU, Basic). So
  raising the tier is not an available workaround there.
- On `msrresrchvc` the A100-80GB **user max is 1 GPU** (both `NC_A100_v4` and
  `NDAMv4`), which is why 2- and 4-GPU jobs there are accepted at submit time and
  then never scheduled. Only A100-40GB (`NDv4`) has a user max above 1 (12), and
  it sits at 381/384 Premium.

### Revised ask

Instead of "restore submission access", the specific request is:

1. Add `sc-vhr286860@microsoft.com` to a group policy on `msrresrchbasicvc` that
   carries `NDH100v5` (and ideally `NDAMv4`) entitlement — the same kind of
   membership that lets `GroupPolicy e9deff52-56e7-4074-bb58-056bbd931bb6` submit
   H100 jobs today.
2. Or confirm which group policy our identity is expected to use now, so we can
   pass it as `-t msrresrchbasicvc:<policy-id>`.
3. If our default-policy submission right was removed deliberately around
   2026-07-26 05:49 UTC, say so, so we stop treating it as an outage.

Jobs admitted before the change are unaffected: our `rq3v2_b2p` H100x4 job on
that VC has been running for three days and is still making progress.
