# 결과 요약 — PD per-layer KV send 프로토타입 (PR #23515 리뷰 지원)

브랜치 `pd-layer-event-send-prototype` (base `67361ff91b`), 프로토타입 코드 ~215줄.
게시: https://github.com/namgyu-youn/sglang/tree/pd-layer-event-send-prototype (포크만, PR 없음)

## 무엇을 만들었나

1. `6bf987a2` — #23515 전송 계층 무변경 적용 (attribution).
2. `af2e9f53` — producer 훅: 일반 prefill forward의 `set_kv_buffer`가 레이어별
   `LayerDoneCounter` 이벤트를 기록하고 그 레이어의 send를 즉시 enqueue
   (cctry의 event-reuse 설계). 플래그 `SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1`
   (기본 off). 모델별 변경 0.
3. `bed85638` — coordinator CPU 유닛 테스트.
4. 벤치마크 드라이버/원본 데이터: `bench/bench_layer_event_send.py`, `bench/bench_flag{0,1}sc.json`

## 검증 상태 (2026-07-05, 2× RTX A6000, mooncake_tcp)

- ✅ 유닛 테스트 7/7.
- ✅ 1P1D e2e: cache-miss 요청은 flag=0 baseline과 bit-exact.
- ❌ radix-cache-hit 요청(cached=2/5/8 모두)은 출력이 결정적으로 달라짐 — 미해결 버그.
- ✅ TTFT 벤치마크: **−12.2%** (아래).

## 1P1D e2e 재현

**prefill에 `--disable-overlap-schedule` 필수** — arming 훅이 non-overlap 루프에만
있어 기본 스케줄러에선 플래그가 조용히 무시됨. 활성 여부는 prefill 로그의
`Layer-event KV send armed`로 확인.

```bash
# GPU 0 — prefill
SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct --trust-remote-code \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 8998 --tp 1 --base-gpu-id 0 --port 30000 \
  --disable-overlap-schedule

# GPU 1 — decode
SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct --trust-remote-code \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 8998 --tp 1 --base-gpu-id 1 --port 30001

# mini-lb 라우터
python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 127.0.0.1 --port 30002

# 요청 (Content-Type 헤더 필수; flag=0으로 재실행해 출력 diff)
curl -s http://127.0.0.1:30002/generate -H "Content-Type: application/json" -d '{
  "text": "The capital of France is",
  "sampling_params": {"temperature": 0, "max_new_tokens": 16}
}' | python3 -m json.tool
```

CUDA toolkit 없는 장비: `pip install mooncake-transfer-engine sglang-router
nvidia-cuda-runtime-cu12 nvidia-cuda-cccl==13.2.75`, `apt install libibverbs1
libnuma1`, `CUDA_HOME=<venv>/nvidia/cu13`(+ `lib64`/`libcudart.so` 심링크),
양쪽 서버에 `--attention-backend triton --sampling-backend pytorch`.

## 벤치마크 결과

~9,470-token 프롬프트 × 10, 단일 청크(`--chunked-prefill-size 16384`),
`--disable-radix-cache`(버그 회피), `max_new_tokens=1` → e2e ≈ TTFT.

| | flag=0 | flag=1 | 차이 |
|---|---|---|---|
| mean | 785.4 ms | **689.6 ms** | **−12.2%** |
| median | 795.7 ms | 688.4 ms | −13.5% |
| min–max | 730.0–813.1 | 675.4–705.3 | 분포 겹침 없음 |

출력은 13개 프롬프트 전부 동일. KV ~300MB/req의 TCP 전송(~100ms)이 compute 뒤에
직렬로 붙던 것이 거의 전부 숨겨진 것. 단, loopback TCP + 1B 모델 + 순차 요청
조건 — RDMA/대형 모델/동시성 재측정 전에는 인용 주의.

## 발견 사항 (리뷰 코멘트용)

- #23515 `_get_pipeline_group_size`의 launch-time 폴백 가드 10개는 대부분
  split-prefill이 일반 run_batch를 우회해서 생기는 제약 — event-reuse는 대부분 회피.
- 두 설계 모두 마지막 청크만 파이프라인 — cctry "<20ms ceiling" 논쟁 지점.
- `LayerDoneCounter`는 HiCache 전용 배선이라 프로토타입이 별도 인스턴스 생성.

## 미해결

- **radix-cache-hit 정확성 버그**: cached prefix 있는 요청의 KV가 잘못 전송됨
  (temperature 0, 100% 재현). 전송 인덱스/이벤트 매핑 문제로 추정, 루트커즈 미확정.
  해결 전엔 성능 주장은 "cache-miss 한정" 조건부.
- overlap 스케줄러 미지원 (기본값에서 플래그 무시됨) — coordinator의 slot-rotation
  안전성이 non-overlap 동기화에 의존, 설계 재검토 필요.
- 하이브리드(Mamba/SWA/DSA) 상태는 `set_kv_buffer`를 안 지나감 — 미구현.
- `run_batch` 예외 시 notifier가 armed로 남음 — 실제 PR엔 try/finally 필요.
- page_first/HiCache 호환은 범위 밖.

## 하지 않은 것

- PR/이슈/리뷰 코멘트 게시 없음 (브랜치는 개인 포크에만).
- #23515 모델 파일 안 건드림. 공개적인 성능 우위 주장 없음.
