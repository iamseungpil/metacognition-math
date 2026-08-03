> ⚠️ **ARCHIVED 2026-08-03 — resolved incident.** The submission block described here is
> over: the RQ3v2-F arms are running on msrresrchbasicvc. Kept as the evidence log for the
> escalation, not as an open issue.

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

## ~~UPDATE 2026-07-27 — the block is a GROUP POLICY membership gap~~ — **SUPERSEDED, see 07-28 below**

> ⚠️ **The ask in this section is withdrawn.** Its reading of the `GroupPolicy` tag is wrong
> (see 07-28). Its *symptom description* — quota is visible while admission fails — still holds
> and is why the section is kept rather than deleted.

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

---

## UPDATE 2026-07-28 — RETRACTION of the 0727 group-policy reading, plus a client-version test that rules out our tooling

### 1. Retraction: `GroupPolicy: e9deff52-...` is the submitter's object ID, not a policy to join

The full job spec of the working H100 job was obtained. In it:

```
tags:      GroupPolicy: e9deff52-56e7-4074-bb58-056bbd931bb6
createdBy: userObjectId: e9deff52-56e7-4074-bb58-056bbd931bb6   ("Woogyeol Jin (SC-ALT)")
userId:                  e9deff52-56e7-4074-bb58-056bbd931bb6
```

The tag value is **identical to the submitting user's AAD object ID**. It is submitter
identity metadata, not a group-policy resource. Asking to be "added to group policy
e9deff52-..." is therefore meaningless — it would amount to asking to become that user.
**The 0727 "Revised ask" item 1 and 2 are withdrawn.** (`amlt cache expand-sku -t
"msrresrchbasicvc:e9deff52-..."` failing is consistent with this: there is no such policy.)

### 2. Client version is ruled out — three versions, same server error

Because the working job ran `amlt.version: 11.14.2` and we had only ever tried 11.9.1 (the
version in use when the block began) and 11.16.0, 11.14.2 was installed into an isolated venv
from the internal index and the same 1-CPU canary was submitted under the **same identity**:

| amlt version | result on `msrresrchbasicvc` |
|---|---|
| 11.9.1 (in use when block began) | `UserError: The virtual cluster does not exist` |
| **11.14.2** (version of the working job) | **`UserError: The virtual cluster does not exist`** |
| 11.16.0 (current) | `UserError: The virtual cluster does not exist` |

The failure surfaces as `HttpResponseError` from `handle_job_submission`, i.e. it is the
**service** rejecting the request, not client-side target resolution. (11.14.2 first reported
"target could not be found" only because the fresh venv had an empty target cache; after
`amlt target list sing` populated it — and it lists `msrresrchbasicvc` with H100 among its
accelerators — the submission produced the same server error.)

**Conclusion: the block is not our tooling, our yaml, our SKU alias, or our client version.
It is an identity/entitlement decision made service-side.**

### 3. Correlation IDs for the support team

From a failing submission at **2026-07-28T06:23:47Z**, region **westus2**:

```
operation: c3840a2450a9283d91b827e4c548679f
request:   c873fcb3948dd25d
```

### 4. Corrected ask

Please look up the admission decision for **`sc-vhr286860@microsoft.com`**
(AAD object id **`a22660cc-8fa9-4dd1-b5fc-eafa9718e257`**) against virtual cluster
`/subscriptions/22da88f6-1210-4de2-a5a3-da4c7c2a1213/resourcegroups/gcr-singularity/providers/microsoft.machinelearningservices/virtualclusters/msrresrchbasicvc`
from workspace `msra-sh-aml-ws`, using the correlation IDs above, and tell us:

1. **Why** the service maps this request to "the virtual cluster does not exist" — which
   entitlement, allowlist, namespace, or Hobo/quota-subscription binding is missing. The
   working job carries `HoboSubscription: cd7aec42-baee-4068-be0f-f41c0f1b8347`; we cannot
   see what ours resolves to.
2. Whether our submission right was **removed deliberately** on 2026-07-26 ~05:49 UTC, so we
   can stop treating this as an outage.
3. If it was not deliberate, please **restore or rebind** it.

Note that jobs admitted before that timestamp were unaffected: our `rq3v2_b2p` H100x4 job kept
running for three more days and completed all 300 steps on 2026-07-28.

### 5. Additional verified evidence (2026-07-28, independent review)

- **Our identity**: `sc-vhr286860@microsoft.com`, AAD object id
  `a22660cc-8fa9-4dd1-b5fc-eafa9718e257` (verified via `az ad signed-in-user show`).
  The working submitter is a different object id, `e9deff52-...`.
- **The VC ARM id we submit is byte-identical to the working job's.** Our Amulet target cache
  holds `/subscriptions/22da88f6-.../resourceGroups/gcr-singularity/providers/Microsoft.MachineLearningServices/virtualclusters/msrresrchbasicvc`,
  the same string as the working job's `azureml.VC`, and Resource Graph shows exactly one VC
  of that name across every subscription visible to us. Stale/incorrect target resolution is ruled out.
- **ARM reads on the VC still succeed for us today.** The per-VC quota GET returns live numbers
  (NDH100v5 Standard 560 total / 218 busy; per-user "Overall" 16) and that per-user row comes from
  `defaultGroupPolicyOverallQuotas` — i.e. the default group policy still *quotes* us while
  managementfrontend refuses to *admit* us.
- **`groupPolicies` on the VC enumerate as 17 objects, each named after an individual user's
  object id — and neither our object id nor the working submitter's exists among them.** A direct
  GET for either returns "Unable to find group policy", symmetrically. This is the concrete reason
  the 07-27 `expand-sku -t "msrresrchbasicvc:e9deff52-..."` probe proved nothing: it fails for every
  object id, including that of a user actively running on the VC.
- **AAD group difference between the two identities is small.** The working submitter holds four
  groups we do not (`bonete06-scalt-esg`, `bonete06-m365`, `UHF Users M365 Group`, and an alphabet
  shard); we hold two they do not (a different alphabet shard, and a conditional-access rollout
  group). All GCR/MSR groups are shared. **No group named after `msrresrchbasicvc` exists in the
  tenant at all**, so there is no obvious "add them to this group" answer visible from our side.
- **Timeline that fits.** The identical error string appeared transiently on 2026-07-15 22:41 during
  preparation for the GCR-wide B200 reallocation; the reallocation executed on 07-16 with new
  allocations going live, and allocation updates were to be handled by each lab's GPU delegate —
  ours was never contacted. The permanent block then began 07-26 05:49 with an unchanged client.

**Leading hypothesis (stated as a hypothesis, not a finding):** the post-reallocation allocation
migration did not carry this identity over, and the legacy admission path was decommissioned at
07-26 05:49. We cannot see the Singularity allocation registry, so we cannot confirm this — which is
precisely what we are asking you to check.

**Earlier correlation IDs** (2026-07-26): operation `cebe26f9d9105d31e359ad11ce058a18`,
request `478560a673cba078`.
