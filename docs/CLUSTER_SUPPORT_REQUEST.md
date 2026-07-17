# CLUSTER_SUPPORT_REQUEST — msrresrchbasicvc 제출 차단 지원 요청 (2026-07-17)

7/16 GCR 전체 유지보수·재할당 이후 우리 워크스페이스의 msrresrchbasicvc 제출만
거절되는 사건의 **지원 요청 문안과 근거 로그**. 보낼 곳: Teams **@gcrsupp**
(또는 "maintenance concluded" 공지 스레드 회신), CC: 랩 GPU delegate,
GPUAllocation@microsoft.com.

## 보낼 메시지 (영문, 그대로 복사)

> **Subject: msrresrchbasicvc submissions rejected with "VC does not exist" since the July-16 reallocation — workspace↔VC association appears dropped**
>
> Hi @gcrsupp,
>
> Since the July 16 maintenance/reallocation, **every job submission from our
> workspace to msrresrchbasicvc fails**, while everything else checks out:
>
> - Error: `(UserError) The virtual cluster does not exist...` from
>   **managementfrontend** (westus2). Latest request id **`585602a56483eb72`**
>   (operation `681b0d4d365604ac9aed9559bfffb186`, 2026-07-17T14:15Z); another
>   sample `450987dc7e84f20f` (13:12Z).
> - Workspace **msra-sh-aml-ws**, project skilldiscovery2, identity
>   v-seungplee@microsoft.com.
> - **Decisive differential test**: the exact same minimal job (echo) submitted
>   from the same workspace/identity is **accepted and runs on msrresrchvc**,
>   but is **rejected on msrresrchbasicvc**. Post-maintenance submissions by
>   colleagues to msrresrchvc also succeed (e.g. run `sweet_nut_mqvdj1bfxm`,
>   created 2026-07-17T08:40Z, running). This isolates the failure to the
>   **msrresrchbasicvc ↔ msra-sh-aml-ws association** specifically.
> - The VC resource exists in ARM (we can enumerate it), quota reads work,
>   code upload / experiment creation succeed — only the final VC mapping is
>   rejected. Reproduced on amlt 11.9.1 and 11.14.2; caches cleared; both
>   `sla_tier: Standard` and `Basic` fail identically. The same launcher YAMLs
>   submitted and ran successfully until July 15 22:41 UTC.
>
> Could you please **restore this workspace's association / allocation
> registration for msrresrchbasicvc (H100)**? Two paper-critical training runs
> are blocked since the maintenance. Thank you!

## 근거 요약 (내부용)

- 차단 시작: 2026-07-15 22:41 UTC. "maintenance concluded·제출 가능" 공지
  (7/17) 이후에도 지속.
- 소거된 가설: yaml 내용(최소 echo yaml도 동일 거절), 인증(amlt는
  v-seungplee로 정상 — 실험 생성·코드 업로드 성공), storage(업로드 단계 통과),
  클라이언트 버전(11.14.2 동일 거절)·캐시·그룹정책 캐시, sla_tier(개명 반영한
  Basic도 동일), B200/타 GPU 대안(타겟 부재 + 매치드 래더 제약).
- 확정 사실: VC는 ARM 실존(구독 22da88f6-…, rg gcr-singularity), 타 팀은
  가동 중(H100 사용량 증가), 같은 워크스페이스에서 msrresrchvc 제출은 됨.
- 부수 확인: 7/16 재할당에서 H100 1120-슬롯 opportunistic 풀의 티어 명칭이
  Standard→**Basic**으로 개편됨(560 풀이 Standard 명칭 인수) — 런처 3종에
  선반영 완료(커밋 91f3878).
- 복구 시: 10분 재시도 루프가 자동 발사(b3pkg fresh + b2 gs140 resume,
  코드 tarball rq3-code-20260717 / asset 480254660).
