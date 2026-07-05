# 결과 요약 — PD per-layer KV send 프로토타입 (PR #23515 리뷰 지원)

브랜치 `pd-layer-event-send-prototype` (base `67361ff91b`), 커밋 3개, 총 ~215줄
(cctry 추정 <150줄 근접).

## 문서 지도

| 문서 | 용도 |
|---|---|
| **이 파일** | 결과 요약 |
| `bench/bench_layer_event_send.py` | TTFT A/B 벤치마크 드라이버 (2026-07-05 작성) |
| `bench/bench_flag{0,1}sc.json` | 벤치마크 원본 데이터 |

(이전 버전이 참조하던 `COMPARISON.md`, `bench/bench_layer_event_send.sh`,
`bench/analyze_layer_event_send.py`는 실제로 존재하지 않아 제거함.)

## 무엇을 만들었나

1. `6bf987a2` — #23515 전송 계층 무변경 적용 (attribution). Transport가
   스케줄러 변경과 분리 가능함을 보여줌.
2. `af2e9f53` — producer 훅: 일반 prefill forward가 `set_kv_buffer` 호출 시
   레이어별로 `LayerDoneCounter` 이벤트를 기록하고 그 레이어의 RDMA send를
   즉시 enqueue (cctry의 event-reuse 설계). 플래그
   `SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1` (기본 off). 모델별 변경 0.
3. `bed85638` — coordinator CPU 유닛 테스트.

## 검증 상태

- ✅ 유닛 테스트 7/7 통과 (CPU, GPU 불필요).
- ✅ GPU 장비 설치 확인 완료 (rust 툴체인 + `uv pip install -e "python[dev]"`).
- ✅ 1P1D 기능 e2e 완료 (2× RTX A6000, `mooncake_tcp`, 2026-07-05):
  - **cache-miss(cached=0) 요청: flag=1 출력이 flag=0 baseline과 bit-exact 일치.**
  - **❌ radix-cache-hit 요청(cached=2/5/8 모두): flag=1 출력이 baseline과 결정적으로
    달라짐 — per-layer 경로가 cached prefix 있는 요청의 KV를 잘못 전송. 아래
    "미해결" 참조.**
- ✅ TTFT 벤치마크 완료 (2026-07-05, 아래 "벤치마크 결과").

## 1P1D e2e 재현 커맨드

`fake` backend는 prefill에서 launch가 막혀 있으므로(assert) 2-GPU +
`mooncake_tcp`(RDMA 불필요) 사용. **prefill에 `--disable-overlap-schedule` 필수** —
arming 훅이 `event_loop_normal_disagg_prefill`에만 있어서 기본(overlap) 스케줄러로는
플래그를 켜도 조용히 일반 chunk 경로로 동작한다. 경로 활성 여부는 prefill 로그의
`Layer-event KV send armed` 라인으로 확인.

```bash
# GPU 0 — prefill  (--disable-overlap-schedule 없으면 layer-event 경로 비활성!)
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

# mini-lb 라우터 (PD는 라우터를 거쳐야 함)
python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 127.0.0.1 --port 30002

# 요청 전송 + 정확성 비교 (SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=0으로 재실행해 출력 diff)
# 주의: 라우터는 Content-Type 헤더 없으면 422 반환 (서버 직접 호출과 다름)
curl -s http://127.0.0.1:30002/generate -H "Content-Type: application/json" -d '{
  "text": "The capital of France is",
  "sampling_params": {"temperature": 0, "max_new_tokens": 16}
}' | python3 -m json.tool
```

환경 메모 (2026-07-05 검증 장비, CUDA toolkit 미설치 + torch cu13):

- `pip install mooncake-transfer-engine sglang-router` +
  `apt install libibverbs1 libnuma1` 필요.
- mooncake wheel은 `libcudart.so.12` 링크 →
  `pip install nvidia-cuda-runtime-cu12` 후 해당 lib 디렉토리를 `LD_LIBRARY_PATH`에.
- flashinfer JIT가 nvcc 요구 + 번들 CCCL이 nvcc 13.2와 비호환 →
  양쪽 서버에 `--attention-backend triton --sampling-backend pytorch` 추가.
- sglang jit_kernel(rope)도 CUDA_HOME 요구 → `pip install nvidia-cuda-cccl==13.2.75`
  후 `CUDA_HOME=<venv>/nvidia/cu13` + `ln -s lib lib64` +
  `ln -s libcudart.so.13 lib/libcudart.so`.

## 벤치마크 결과 (2026-07-05)

셋업: 2× RTX A6000, `mooncake_tcp`(loopback TCP), Llama-3.2-1B, triton attention
backend, `--disable-radix-cache`(cache-hit 버그 회피) + `--chunked-prefill-size
16384`(단일 청크 = 전체 prefill이 layer-pipelined), 프롬프트 ~9,470 tokens ×
13개(warmup 3 + 측정 10), `max_new_tokens=1` → e2e ≈ TTFT. 드라이버:
`bench/bench_layer_event_send.py`.

| | flag=0 (chunk 전송) | flag=1 (per-layer) | 차이 |
|---|---|---|---|
| mean | 785.4 ms | **689.6 ms** | **−95.8 ms (−12.2%)** |
| median | 795.7 ms | 688.4 ms | −107.3 ms (−13.5%) |
| stdev | 27.1 ms | 8.5 ms | |
| min–max | 730.0–813.1 | 675.4–705.3 | 분포 겹침 없음 |

- flag=1 최대(705ms) < flag=0 최소(730ms) — 별도 통계 검정 불필요한 수준.
- 13개 프롬프트 전부 첫 토큰 출력 동일 (cache-miss 경로, 정확성 유지 확인).
- 해석: KV ~300MB/req(9.5k tok × 32KB/tok)의 loopback TCP 전송 ~100ms가 prefill
  compute(~690ms) 뒤에 직렬로 붙던 것이 거의 전부 겹쳐져 숨겨짐. RDMA 환경에선
  전송이 더 빨라 절대 이득은 줄지만, cctry의 "<20ms ceiling" 논쟁과 달리
  전송량이 큰 모델/长프롬프트에서 의미 있는 TTFT 개선 여지를 보여줌.
- 주의: loopback TCP + 1B 모델 + 단일 요청 순차 측정이라는 제한적 조건. 멀티
  요청 동시성, RDMA, 대형 모델에서의 재측정 필요.

## 발견 사항 (리뷰 코멘트에 쓸 만한 것)

- #23515의 `_get_pipeline_group_size`에는 launch-time 폴백 가드가 10개 — 대부분
  split-prefill이 일반 run_batch를 우회해서 생기는 제약. event-reuse는 이를
  대부분 회피함.
- 두 설계 모두 chunked prefill 중간 청크는 일반 경로, 마지막 청크만 파이프라인
  — cctry의 "<20ms ceiling" 논쟁이 정확히 이 지점.
- `LayerDoneCounter`는 `HiCacheController.__init__`에서 배선됨 — HiCache
  꺼진 서버엔 counter가 없어, 프로토타입이 별도 인스턴스를 만듦.

## 미해결 / 정직하게 남긴 것

- **❌ radix-cache-hit 정확성 버그 (e2e에서 발견, 2026-07-05)**: prefill이 radix
  cache에 prefix가 있는 요청을 받으면 flag=1 출력이 flag=0 baseline과 달라짐
  (temperature 0, 100% 재현; cached=2/5/8 모두 발생, cached=0은 bit-exact).
  cached prefix 페이지 자체는 GPU에 유효한 데이터이므로 단순 race가 아니라
  전송 인덱스/이벤트 매핑 문제로 추정 — 후보: (a) cache-hit 시 forward가 새 토큰만
  쓰는데 이벤트-레이어 매핑이 어긋남, (b) 비연속 페이지(공유 radix 페이지 + 새 할당
  페이지)의 src/dst 블록 그룹핑, (c) `_prepare_layer_send_indices`의 curr_idx 가정.
  루트커즈 미확정. 이 상태로는 프로토타입 주장 불가 — 리뷰 코멘트에 명시 필요.
- 프로토타입 arming 훅이 non-overlap 프리필 루프에만 있음 — overlap 스케줄러(기본값)
  에선 플래그가 조용히 무시됨. coordinator의 slot-rotation 안전성 주장 자체가
  non-overlap 동기화에 의존하므로 overlap 지원은 설계 재검토 필요.
- 하이브리드(Mamba/SWA/DSA) 상태는 `set_kv_buffer`를 안 지나감 — 미구현,
  `COMPARISON.md`에 명시.
- `run_batch` 예외 시 notifier가 armed로 남음 — 실제 PR엔 try/finally 필요.
- page_first/HiCache 호환(쟁점 1)은 범위 밖.

## 하지 않은 것

- GitHub에 아무것도 안 올림. #23515 모델 파일 안 건드림. 성능 우위 주장 없음.
