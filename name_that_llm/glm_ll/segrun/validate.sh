#!/usr/bin/env bash
# Validate ntl-segrun: a GLM5-Next model run by layer segments must give the same scores as run whole.
#   src/segrun/validate.sh BUILD_BIN_DIR PYTHON [NGL] [ARCH=glm5-next|glm-dsa]
# Builds a random 12-layer model of ARCH (test-llama-archs, indexer top-k 64: above the longest text, below the
# KV cells of a unified cache with several texts), cuts it into 3 segments
# (slice_gguf.py), runs the whole model and the segment pipeline with f32/f16/bf16 states and seq-batch 1 and 4,
# then compares (compare.py).
set -euo pipefail
BIN=${1:?build bin dir}
PY=${2:?python}
NGL=${3:-0}
ARCH=${4:-glm5-next}
HERE=$(cd "$(dirname "$0")" && pwd)
T=${NTL_TEST_DIR:-$HERE/../../outputs/segrun_test}/$ARCH
rm -rf "$T"; mkdir -p "$T"
NTL_TEST_GLM5_NLAYER=12 NTL_TEST_INDEXER_TOPK=${NTL_TEST_INDEXER_TOPK:-64} "$BIN/test-llama-archs" -a "$ARCH" -o "$T" > "$T/gen.log" 2>&1
"$PY" -I "$HERE/slice_gguf.py" "$T/$ARCH-moe.gguf" "$T/segs" 0,4,8,12 > "$T/slice.log"
"$PY" -I -c "
import sys; sys.path.insert(0, '$HERE'); import segio, numpy as np
rng = np.random.default_rng(0)
seqs = [(rng.integers(0, 128, size=int(L)).tolist(), int(p)) for L, p in zip(rng.integers(8, 60, size=16), rng.integers(1, 6, size=16))]
segio.write_tokens('$T/tok.bin', seqs)
"
run() { "$BIN/ntl-segrun" run --ngl "$NGL" "$@" >> "$T/run.log" 2>&1 || { echo "FAILED: $*"; tail -30 "$T/run.log"; exit 1; }; }
S=$T/segs
for B in 1 4; do
  run --model "$T/$ARCH-moe.gguf" --tokens "$T/tok.bin" --out "$T/scores_full_b$B.bin" --seq-batch $B
  for dt in f32 f16 bf16; do
    run --model "$S/seg00-of-03-L0-4.gguf"  --tokens "$T/tok.bin" --out "$T/st1.bin" --state-dtype $dt --seq-batch $B
    run --model "$S/seg01-of-03-L4-8.gguf"  --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/st2.bin" --state-dtype $dt --seq-batch $B
    run --model "$S/seg02-of-03-L8-12.gguf" --tokens "$T/tok.bin" --state-in "$T/st2.bin" --out "$T/scores_seg_${dt}_b$B.bin" --seq-batch $B
  done
  # middle segment rewriting its input state in place
  run --model "$S/seg00-of-03-L0-4.gguf"  --tokens "$T/tok.bin" --out "$T/st1.bin" --state-dtype f16 --seq-batch $B
  run --model "$S/seg01-of-03-L4-8.gguf"  --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/st1.bin" --state-dtype f16 --seq-batch $B
  run --model "$S/seg02-of-03-L8-12.gguf" --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/scores_seg_f16inplace_b$B.bin" --seq-batch $B
done
# packed decodes (padding, unified KV, token budget) with half-size ubatches, whole and segmented
run --model "$T/$ARCH-moe.gguf" --tokens "$T/tok.bin" --out "$T/scores_full_pack_b4.bin" --seq-batch 4 --pack --ubatch-div 2
run --model "$S/seg00-of-03-L0-4.gguf"  --tokens "$T/tok.bin" --out "$T/st1.bin" --state-dtype f32 --seq-batch 4 --pack --ubatch-div 2
run --model "$S/seg01-of-03-L4-8.gguf"  --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/st1.bin" --state-dtype f32 --seq-batch 4 --pack --ubatch-div 2
run --model "$S/seg02-of-03-L8-12.gguf" --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/scores_seg_f32pack_b4.bin" --seq-batch 4 --pack --ubatch-div 2
# two halves of each segment on two devices (--split-mid): same scores, and with >= 2 GPUs the layers of the
# middle segment must really be split (first half on CUDA0, second half on CUDA1)
: > "$T/split.log"
runs() { "$BIN/ntl-segrun" run --ngl "$NGL" --split-mid "$@" >> "$T/split.log" 2>&1 || { echo "FAILED: $*"; tail -30 "$T/split.log"; exit 1; }; }
runs --model "$S/seg00-of-03-L0-4.gguf"  --tokens "$T/tok.bin" --out "$T/st1.bin" --state-dtype f32 --seq-batch 4
: > "$T/split_mid.log"
"$BIN/ntl-segrun" run --ngl "$NGL" --split-mid --model "$S/seg01-of-03-L4-8.gguf" --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/st2.bin" --state-dtype f32 --seq-batch 4 > "$T/split_mid.log" 2>&1 || { echo "FAILED split mid"; tail -30 "$T/split_mid.log"; exit 1; }
runs --model "$S/seg02-of-03-L8-12.gguf" --tokens "$T/tok.bin" --state-in "$T/st2.bin" --out "$T/scores_seg_f32split_b4.bin" --seq-batch 4
# the configuration of the real runs: split + pack + half ubatches (pipeline parallel on two GPUs)
runs --model "$S/seg00-of-03-L0-4.gguf"  --tokens "$T/tok.bin" --out "$T/st1.bin" --state-dtype f32 --seq-batch 4 --pack --ubatch-div 2
runs --model "$S/seg01-of-03-L4-8.gguf"  --tokens "$T/tok.bin" --state-in "$T/st1.bin" --out "$T/st2.bin" --state-dtype f32 --seq-batch 4 --pack --ubatch-div 2
runs --model "$S/seg02-of-03-L8-12.gguf" --tokens "$T/tok.bin" --state-in "$T/st2.bin" --out "$T/scores_seg_f32splitpack_b4.bin" --seq-batch 4 --pack --ubatch-div 2
grep "split-mid:" "$T/split_mid.log"
N_GPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c "^GPU " || true)
if [ "$NGL" != "0" ] && [ "$N_GPU" -ge 2 ]; then
  "$PY" -I -c "
import re, sys
dev = dict((int(l), d) for l, d in re.findall(r'layer +(\d+) assigned to device (\w+)', open('$T/split_mid.log').read()))
got = [dev.get(i) for i in range(4, 8)]
print('split-mid devices of layers 4..7:', got)
sys.exit(0 if got == ['CUDA0', 'CUDA0', 'CUDA1', 'CUDA1'] else 1)
" || { echo "split-mid did not split the layers"; exit 1; }
fi
"$PY" -I "$HERE/compare.py" "$T"
