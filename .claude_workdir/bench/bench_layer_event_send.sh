#!/usr/bin/env bash
# Benchmark driver: baseline vs event-reuse per-layer KV send (PR #23515 review).
#
# Targets the disputed "<20ms TTFT ceiling" claim: sweeps chunked-prefill size
# and input length, records TTFT from bench_serving, and captures both server
# logs so analyze_layer_event_send.py can separate prefill-side transfer time
# from decode-side poll-loop latency.
#
# NOT run in the authoring environment (no GPU / no RDMA there). Review the
# host-specific block below before running. Single-node loopback (default)
# measures scheduling/poll overhead but NOT real wire time — for wire time you
# need two nodes (or at least two NICs) with RDMA/RoCE.
#
# Usage:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct bash bench_layer_event_send.sh
#
# Modes compared:
#   baseline    — normal end-of-prefill chunk transfer
#   event-send  — SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1 (this branch)
# To also compare #23515 itself, check out its branch and add a third mode
# with SGLANG_ENABLE_PIPELINED_KV_TRANSFER=1.

set -euo pipefail

########################### host-specific config ############################
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
HOST_IP="${HOST_IP:-127.0.0.1}"          # use the RDMA-reachable IP on 2-node
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
LB_PORT="${LB_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
IB_DEVICE="${IB_DEVICE:-}"               # e.g. mlx5_0; empty lets mooncake pick
ISL_LIST=(${ISL_LIST:-4096 16384 32768})
CHUNKED_PREFILL_LIST=(${CHUNKED_PREFILL_LIST:-2048 4096 8192})
NUM_PROMPTS="${NUM_PROMPTS:-32}"
OUT_DIR="${OUT_DIR:-$(dirname "$0")/results/$(date +%Y%m%d-%H%M%S)}"
##############################################################################

mkdir -p "$OUT_DIR"
echo "results -> $OUT_DIR"

COMMON_ARGS=(
  --model-path "$MODEL"
  --host "$HOST_IP"
  --page-size 64
  --disaggregation-transfer-backend mooncake
  --disable-radix-cache          # keep prefix cache from hiding transfer size
)
[[ -n "$IB_DEVICE" ]] && COMMON_ARGS+=(--disaggregation-ib-device "$IB_DEVICE")

wait_healthy() {
  local port=$1 tries=120
  until curl -sf "http://$HOST_IP:$port/health" >/dev/null; do
    ((tries--)) || { echo "server on :$port never became healthy" >&2; return 1; }
    sleep 5
  done
}

run_one() {
  local mode=$1 cps=$2 isl=$3
  local tag="${mode}_cps${cps}_isl${isl}"
  local envs=()
  [[ "$mode" == "event-send" ]] && envs+=(SGLANG_ENABLE_LAYER_EVENT_KV_TRANSFER=1)
  [[ "$mode" == "pr23515" ]] && envs+=(SGLANG_ENABLE_PIPELINED_KV_TRANSFER=1)

  echo "=== $tag ==="

  env "${envs[@]}" CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    python -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --port "$PREFILL_PORT" \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
      --chunked-prefill-size "$cps" \
      --disable-overlap-schedule \
      > "$OUT_DIR/prefill_$tag.log" 2>&1 &
  local prefill_pid=$!

  CUDA_VISIBLE_DEVICES=$DECODE_GPU \
    python -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --port "$DECODE_PORT" \
      --disaggregation-mode decode \
      > "$OUT_DIR/decode_$tag.log" 2>&1 &
  local decode_pid=$!

  python -m sglang.srt.disaggregation.launch_lb \
    --prefill "http://$HOST_IP:$PREFILL_PORT" \
    --decode "http://$HOST_IP:$DECODE_PORT" \
    --host 0.0.0.0 --port "$LB_PORT" \
    > "$OUT_DIR/lb_$tag.log" 2>&1 &
  local lb_pid=$!

  wait_healthy "$PREFILL_PORT" && wait_healthy "$DECODE_PORT"

  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "http://$HOST_IP:$LB_PORT" \
    --dataset-name random \
    --random-input-len "$isl" \
    --random-output-len 8 \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate 1 \
    --output-file "$OUT_DIR/bench_$tag.jsonl" \
    > "$OUT_DIR/bench_$tag.log" 2>&1 || echo "bench failed for $tag" >&2

  kill "$lb_pid" "$decode_pid" "$prefill_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  sleep 5
}

for mode in baseline event-send; do
  for cps in "${CHUNKED_PREFILL_LIST[@]}"; do
    for isl in "${ISL_LIST[@]}"; do
      run_one "$mode" "$cps" "$isl"
    done
  done
done

python "$(dirname "$0")/analyze_layer_event_send.py" "$OUT_DIR"
