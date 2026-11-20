"""Build the Kaggle notebook that scores train.csv under the real GLM-5.3 and GLM-5.3-Flash.

Two of the 18 labels are open-weight models, and they are the blend's weakest pair. José Padilla's
public ``ntl-segrun`` pipeline (see ``name_that_llm/glm_ll/``) runs both models on two T4s, one
layer segment at a time, and writes 24 likelihood features per text and model. This notebook runs
it over train.csv, so the gain of a likelihood member can be measured on our own out-of-fold blend
(``modeling/glm_blend.py``) before it goes into a submission.

    uv run python -m name_that_llm.kaggle_glm_features                  # MODE = "full"
    uv run python -m name_that_llm.kaggle_glm_features --mode timing    # first 40 passages
    uv run python -m name_that_llm.kaggle_glm_features --push-dir <dir> --user <kaggle user>
    uv run python -m name_that_llm.kaggle_glm_features --tpu --mode timing   # TPU VM, CPU only

On Kaggle (GPU T4 x2, internet off) it needs the competition data, the three segment datasets in
``SEGMENT_DATASETS`` and the output of the notebook ``BINARY_NOTEBOOK``. ``--push-dir`` writes the
notebook with a ``kernel-metadata.json`` that attaches them, ready for ``kaggle kernels push``.

``--tpu`` builds the alternative for a Kaggle TPU VM, which spends TPU quota instead of GPU quota.
llama.cpp cannot use the TPU chips, so that notebook builds ``ntl-segrun`` for the VM's host CPUs
from the author's patch (``glm_ll/segrun/``, internet on to clone llama.cpp), checks the build with
the author's segmented-vs-whole validation, and runs the same pipeline with one thread per core.
Its ``probe`` mode only checks the machine and the inputs; ``timing`` scores the same 40 passages
as the GPU timing run (compare them with ``modeling/glm_compare.py``) and forecasts the full run
from ``ntl-segrun``'s own load and compute times; ``SHARDS`` runs a full run as parts in order,
each saved as soon as it finishes.
"""

import base64
import hashlib
import json
from pathlib import Path
import re

import nbformat as nbf
import pandas as pd
import typer

from name_that_llm.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer()

NOTEBOOK_PATH = PROJ_ROOT / "notebooks" / "kaggle_glm_likelihood_train.ipynb"
TPU_NOTEBOOK_PATH = PROJ_ROOT / "notebooks" / "kaggle_glm_likelihood_train_tpu.ipynb"
GLM_LL_DIR = Path(__file__).parent / "glm_ll"
# exact segment files and sizes, as pinned by the public 0.702 notebook (lightningv08)
SEGMENTS = {
    "jamesliebel/glm53-iq3xxs-segs-a-mirror": [
        ["glm53_iq3xxs-seg00-of-19-L0-6.gguf", 12753355264],
        ["glm53_iq3xxs-seg01-of-19-L6-10.gguf", 15651714688],
        *[
            [f"glm53_iq3xxs-seg{i:02d}-of-19-L{4 * i + 2}-{4 * i + 6}.gguf", 14583346912]
            for i in range(2, 10)
        ],
    ],
    "jamesliebel/glm53-iq3xxs-segs-b-mirror": [
        *[
            [f"glm53_iq3xxs-seg{i:02d}-of-19-L{4 * i + 2}-{4 * i + 6}.gguf", 14583346912]
            for i in range(10, 18)
        ],
        ["glm53_iq3xxs-seg18-of-19-L74-78.gguf", 16345433920],
    ],
    "jamesliebel/glm53-flash-iq3xxs-segs-mirror": [
        ["flash_iq3xxs-seg00-of-06-L0-8.gguf", 14879704352],
        ["flash_iq3xxs-seg01-of-06-L8-16.gguf", 22864262976],
        ["flash_iq3xxs-seg02-of-06-L16-24.gguf", 21821175680],
        ["flash_iq3xxs-seg03-of-06-L24-32.gguf", 21821175680],
        ["flash_iq3xxs-seg04-of-06-L32-40.gguf", 21821175680],
        ["flash_iq3xxs-seg05-of-06-L40-45.gguf", 14407253632],
    ],
}
SEGMENT_DATASETS = list(SEGMENTS)
BINARY_NOTEBOOK = "estopadilla/ntl-glm-likelihood-train"
# the prompt prefix and the 160-token cut the public likelihood head was fitted with
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>"
MAX_TEXT_TOKENS = 160
TIMING_ROWS = 40
# the author's ntl-segrun sources (glm_ll/segrun/, copied unchanged), checked before they are embedded
LLAMA_TAG = "b12247"
SEGRUN_FILES = {
    "llama-cpp-b12247-ntl.patch": "98d54fd81f292c605376ca6d106dfe7fa401bf367d4a8b29e83e1ab55837c8cd",
    "validate.sh": "6ef51f6f890b44054436a64f6cb6290a4f2bad531bfc850f978dc3141f49a43d",
    "slice_gguf.py": "af4d5b98486334b557588da12de9f55c6672fc6767f483521c18fea0d3feeb86",
    "compare.py": "591ddf0600744b16372502090158c8a43661f319f739df96949301ceac900cf0",
}
TPU_MACHINE_SHAPE = "TpuV5E8"

SETUP_CODE = """
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

WORK = Path('/kaggle/temp/glm_ll')
OUT = Path('/kaggle/working/glm_ll')
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f'nothing matches {pattern}: is the input attached?'
    return hits[0]


TRAIN = find_one('/kaggle/input/**/name-that-llm/train.csv')
FLASH = sorted(glob.glob('/kaggle/input/**/flash_iq3xxs-seg*.gguf', recursive=True))
GLM53 = sorted(glob.glob('/kaggle/input/**/glm53_iq3xxs-seg*.gguf', recursive=True))
assert len(FLASH) == 6 and len(GLM53) == 19, (len(FLASH), len(GLM53))

artifacts = Path(find_one('/kaggle/input/**/ntl-bin/ntl-segrun')).parent
for name in ('lib/libcublas.so.13', 'lib/libcublasLt.so.13', 'lib/libcudart.so.13'):
    assert (artifacts / name).is_file(), artifacts / name
BIN = Path('/kaggle/temp/glm_ll_bin')
BIN.mkdir(exist_ok=True)
shutil.copyfile(artifacts / 'ntl-segrun', BIN / 'ntl-segrun')
(BIN / 'ntl-segrun').chmod(0o755)
# the binary was built against CUDA 13; its runtime ships next to it
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{artifacts / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)
print(f'{len(pd.read_csv(TRAIN))} train passages; segments and binary found')
"""

RUN_CODE = """
limit = TIMING_ROWS if MODE == 'timing' else -1
for tag, segs in (('flash', FLASH), ('glm53', GLM53)):
    t0 = time.time()
    cmd = [sys.executable, str(WORK / 'run_pipeline.py'), '--bin', str(BIN), '--segs', json.dumps(segs),
           '--csv', TRAIN, '--out', str(OUT), '--tag', tag, '--work', str(WORK / 'state'),
           '--seq-batch', '8', '--state-dtype', 'auto', '--max-text-tokens', str(MAX_TEXT_TOKENS),
           '--prefix', PREFIX, '--limit', str(limit)]
    subprocess.run(cmd, env=ENV, check=True)
    timing = json.loads((OUT / f'{tag}_timing.json').read_text())
    print(f"{tag}: {timing['n_done']} texts, {timing['n_rows']} token rows, "
          f"{len(timing['chunks'])} chunk(s), {time.time() - t0:.0f} s", flush=True)

feats = {t: pd.read_csv(OUT / f'{t}_features.csv') for t in ('flash', 'glm53')}
assert list(feats['flash'].id) == list(feats['glm53'].id)
print(feats['glm53'].describe().T[['mean', 'std']].round(3))
"""

# Forecast of a full run from a timing run. ntl-segrun prints "done. load X s, compute Y s" for every
# segment: loading does not grow with the number of texts, computing does (per token row).
FORECAST_CODE = """
import re


def forecast(tag, n_texts_full):
    # (load s, compute s) of a full run of one model, from a timing run's log
    timing = json.loads((OUT / f'{tag}_timing.json').read_text())
    log = (OUT / f'{tag}_run.log').read_text(errors='replace')
    done = [(float(a), float(b)) for a, b in re.findall(r'done\\. load ([0-9.]+) s, compute ([0-9.]+) s', log)]
    assert done, f'no per-segment timings in {tag}_run.log'
    compute = sum(b for _, b in done)
    # everything but computing stays fixed: loading, process start, state I/O, tokenizing (wall time,
    # which runs 16-27% above the "done." lines)
    fixed = timing['total_s'] - compute
    rows_full = timing['n_rows'] / timing['n_done'] * n_texts_full
    compute_full = compute * rows_full / timing['n_rows']
    print(f"{tag}: {len(done)} segments, fixed {fixed / 60:.1f} min, compute {compute / 60:.1f} min for "
          f"{timing['n_rows']} rows ({timing['n_rows'] * len(done) / max(compute, 1e-9):.0f} rows/s per segment). "
          f"Full run: about {rows_full:.0f} rows, {(fixed + compute_full) / 3600:.2f} h", flush=True)
    return fixed, compute_full
"""


def pipeline_source(name: str) -> str:
    """A pipeline file as text (run_pipeline.py imports its siblings by bare name, so no import here)."""
    return (GLM_LL_DIR / f"{name}.py").read_text(encoding="utf-8")


def segrun_payload() -> dict[str, str]:
    """The author's ntl-segrun files as base64, after checking each one's sha256 (LF line endings)."""
    out = {}
    for name, sha in SEGRUN_FILES.items():
        raw = (GLM_LL_DIR / "segrun" / name).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        assert got == sha, f"glm_ll/segrun/{name}: sha256 {got} != {sha} (CRLF line endings?)"
        out[name] = base64.b64encode(raw).decode("ascii")
    return out


def indent(code: str, prefix: str = "    ") -> str:
    return "".join(prefix + line + "\n" if line else "\n" for line in code.strip().split("\n"))


def finish(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def build(mode: str = "full") -> nbf.NotebookNode:
    assert mode in ("timing", "full")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            "# Name That LLM: GLM-5.3 likelihood features for train.csv\n\n"
            "Scores every train passage under the real GLM-5.3 and GLM-5.3-Flash weights and writes "
            "24 likelihood features per passage and model to `glm_ll/`. "
            "Needs GPU T4 x2 and no internet.\n\n"
            "The scoring pipeline (`ntl-segrun`, `run_pipeline.py`, `features.py`, `segio.py`) is "
            "José Padilla's, from *0.689 | Real GLM-5.3 Likelihood on 2xT4*; the segment GGUFs are "
            "jamesliebel's mirrors of jinghengwu's quantization of Z.ai's open weights. "
            "This notebook only runs them over train.csv."
        ),
        code(
            f"MODE = {mode!r}  # 'timing': the first {TIMING_ROWS} passages; 'full': all of train.csv\n"
            f"PREFIX = {PREFIX!r}\n"
            f"MAX_TEXT_TOKENS = {MAX_TEXT_TOKENS}\n"
            f"TIMING_ROWS = {TIMING_ROWS}\n" + SETUP_CODE
        ),
    ]
    for name in ("segio", "features", "run_pipeline"):
        cells.append(code(f"%%writefile /kaggle/temp/glm_ll/{name}.py\n" + pipeline_source(name)))
    cells.append(code(RUN_CODE.strip() + "\n"))
    cells.append(
        code(
            FORECAST_CODE.strip()
            + "\n\n\nif MODE == 'timing':\n"
            + "    for tag in ('flash', 'glm53'):\n"
            + "        forecast(tag, len(pd.read_csv(TRAIN)))\n"
        )
    )
    return finish(cells)


# ---------------------------------------------------------------- TPU VM (host CPUs only)

TPU_SETUP_CODE = """
import base64
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

T0 = time.time()


def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-6000:], r.stderr[-3000:], flush=True)
    if check:
        assert r.returncode == 0, f'failed ({r.returncode}): {cmd}'
    return r


# ---- the machine: a TPU v5e-8 VM has 224 vCPUs (2 x 56-core AMD EPYC); a push can fall back to 4 ----
sh('nproc; lscpu | head -30; free -g; df -h / /kaggle/working /kaggle/temp /tmp /dev/shm 2>/dev/null; '
   'cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null; ls /dev/accel* /dev/vfio 2>/dev/null; '
   'mount | grep /kaggle/input | head -3; which git gcc g++ cmake make; gcc --version | head -1', check=False)
print('image', os.environ.get('KAGGLE_DOCKER_IMAGE'), flush=True)
assert len(os.sched_getaffinity(0)) >= 200, (
    f'{len(os.sched_getaffinity(0))} CPUs: not a TPU VM host (did the push fall back to a CPU machine?)')
cpu = sh('lscpu -p=CORE,SOCKET', check=False).stdout
PHYSICAL = len({line for line in cpu.splitlines() if line and not line.startswith('#')}) or os.cpu_count()
FLAGS = set(re.search(r'^flags\\s*:(.*)$', open('/proc/cpuinfo').read(), re.M).group(1).split())
print(f'{os.cpu_count()} logical CPUs, {PHYSICAL} physical cores; AVX2 {"avx2" in FLAGS}, '
      f'AVX-512 {"avx512f" in FLAGS}, AVX-512 VNNI {"avx512_vnni" in FLAGS}, AMX {"amx_tile" in FLAGS}')
assert {'avx2', 'fma', 'f16c'} <= FLAGS, 'no AVX2/FMA/F16C: ggml would be very slow here'
mem = int(re.search(r'MemAvailable:\\s+(\\d+) kB', open('/proc/meminfo').read()).group(1)) / 2**20
print(f'{mem:.0f} GiB RAM available', flush=True)
assert mem >= 100, 'too little RAM to keep a segment in the page cache'
# No OMP_PROC_BIND/OMP_PLACES: libgomp would pin ntl-segrun's main thread to one core, and the
# per-text scoring threads it starts (std::thread, not OpenMP) would all inherit that core.
THREADS = THREADS or PHYSICAL

# work folder: the local disk with the most free space (the state between segments goes there;
# never /kaggle/working, which is ~20 GB and becomes the saved output)
roots = [Path(p) for p in ('/kaggle/temp', '/tmp', '/') if os.path.isdir(p) and os.access(p, os.W_OK)]
ROOT = max(roots, key=lambda p: shutil.disk_usage(p).free)
WORK = ROOT / 'glm_ll'                 # passage lists and the state between segments
BASE = Path('/tmp/ntl')                # validate.sh expects src/segrun and third_party/llama.cpp side by side
SRC = BASE / 'src' / 'segrun'
LLAMA = BASE / 'third_party' / 'llama.cpp'
BIN = BASE / 'bin'
OUT = Path('/kaggle/working/glm_ll')
for d in (WORK, SRC, BIN, OUT):
    d.mkdir(parents=True, exist_ok=True)
print('work folder', WORK, f'{shutil.disk_usage(ROOT).free / 2**30:.0f} GiB free', flush=True)
# the f32 state of a full Flash run is 22.9 GB (GLM-5.3: 8.6 GB); less room means more chunks, and every
# chunk reads all the segments again
if shutil.disk_usage(ROOT).free < 35e9:
    print('WARNING: under 35 GB free for the state: a full Flash run will need several chunks', flush=True)

# ---- the inputs: every segment present, with its exact size ----
TRAIN = sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True))
assert TRAIN, 'competition data not attached'
TRAIN = TRAIN[0]
FLASH, GLM53 = [], []
for ref, files in SEGMENTS.items():
    for name, size in files:
        hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
        assert hits, f'{name} missing: attach {ref}'
        got = os.path.getsize(hits[0])
        assert got == size, f'{hits[0]}: {got} bytes, expected {size}'
        (FLASH if name.startswith('flash_') else GLM53).append(hits[0])
FLASH.sort()
GLM53.sort()
assert len(FLASH) == 6 and len(GLM53) == 19
print(f'{len(pd.read_csv(TRAIN))} train passages; 25 segments found with the right sizes '
      f'({sum(os.path.getsize(p) for p in FLASH + GLM53) / 1e9:.0f} GB)', flush=True)
# read speed of the inputs (4 GiB of one segment; the first read of each segment is a cold one)
sh(f'dd if={FLASH[1]} of=/dev/null bs=16M count=256 2>&1 | tail -1', check=False)
"""

TPU_BUILD_CODE = """
# ---- the author's sources: checked, then llama.cpp at the tag the patch was written for ----
for name, sha in SEGRUN_SHA256.items():
    raw = base64.b64decode(SEGRUN_B64[name])
    assert hashlib.sha256(raw).hexdigest() == sha, name
    (SRC / name).write_bytes(raw)
sh(f'{sys.executable} -I -c "import numpy, yaml"')   # validate.sh runs slice_gguf.py (gguf-py) this way
if not (LLAMA / '.git').exists():
    sh(f'git clone -q --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp {LLAMA}')
    sh(f'cd {LLAMA} && git apply --check {SRC}/llama-cpp-b12247-ntl.patch && '
       f'git apply {SRC}/llama-cpp-b12247-ntl.patch && git status --short | head -20')
assert (LLAMA / 'gguf-py' / 'gguf').is_dir(), 'llama.cpp clone failed (internet on?)'

# ---- build ntl-segrun for this CPU: no CUDA, native instruction set, OpenMP threads ----
# always name the targets: a bare build also links test-chat/test-server, which need the server
t = time.time()
if shutil.which('cmake') is None:
    sh(f'{sys.executable} -m pip install -q cmake ninja')
CMAKE_FLAGS = ('-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=ON '
               '-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF '
               '-DLLAMA_BUILD_EXAMPLES=OFF')
sh(f'cd {LLAMA} && cmake -S . -B build-cpu {CMAKE_FLAGS} > build.log 2>&1 || (tail -40 build.log; false)')
sh(f'cd {LLAMA} && cmake --build build-cpu -j{os.cpu_count()} --target ntl-segrun test-llama-archs >> build.log 2>&1 '
   '|| (grep -n -i "error" build.log | head -40; false)')
for name in ('ntl-segrun', 'test-llama-archs'):
    shutil.copy(LLAMA / 'build-cpu' / 'bin' / name, BIN / name)
    (BIN / name).chmod(0o755)
deps = sh(f'ldd {BIN}/ntl-segrun', check=False).stdout
assert not re.search(r'libcu(da|blas|dart)', deps), 'the CPU build links CUDA libraries'
print(f'build {time.time() - t:.0f} s', flush=True)

# ---- the author's check: a random model run by layer segments scores the same as run whole ----
t = time.time()
for arch in ('glm5-next', 'glm-dsa'):
    r = sh(f'NTL_TEST_DIR={BASE}/validate bash {SRC}/validate.sh {BIN} {sys.executable} 0 {arch}', check=False)
    assert r.returncode == 0 and 'ALL OK' in r.stdout, f'validation failed for {arch}'
print(f'validation {time.time() - t:.0f} s', flush=True)

# a record of what was built, kept with the output
(OUT / 'ntl-bin').mkdir(exist_ok=True)
shutil.copy(BIN / 'ntl-segrun', OUT / 'ntl-bin' / 'ntl-segrun')
model_name = re.search(r'^model name\\s*:(.*)$', open('/proc/cpuinfo').read(), re.M).group(1).strip()
(OUT / 'ntl-bin' / 'stamp.json').write_text(json.dumps({
    'llama_tag': LLAMA_TAG,
    'llama_commit': sh(f'git -C {LLAMA} rev-parse HEAD', check=False).stdout.strip(),
    'patch_sha256': SEGRUN_SHA256['llama-cpp-b12247-ntl.patch'],
    'cmake_flags': CMAKE_FLAGS,
    'cpu': model_name,
    'cpu_flags_sha256': hashlib.sha256(' '.join(sorted(FLAGS)).encode()).hexdigest(),
    'ntl_segrun_sha256': hashlib.sha256((BIN / 'ntl-segrun').read_bytes()).hexdigest(),
    'validate': 'ALL OK (glm5-next, glm-dsa)',
}, indent=1))
"""

TPU_RUN_CODE = """
# ---- score the passages: one model at a time, every segment once per state chunk ----
train = pd.read_csv(TRAIN)
if MODE == 'timing':
    jobs = [(train.iloc[:TIMING_ROWS], '')]     # the same passages as the GPU timing run
else:
    # the parts in order; each one's outputs are written as soon as it finishes, so a session that
    # ends early keeps the parts it completed (a model on its own writes nothing until its end)
    jobs = [(train.iloc[k::n], '' if n == 1 else f'_part{k}of{n}') for k, n in SHARDS]
SETUP_S = time.time() - T0       # checks, build and validation: paid again by every session
(OUT / 'setup.json').write_text(json.dumps({'setup_s': SETUP_S}))

stop = False
for part, suffix in jobs:
  if stop:
    break
  csv = WORK / f'texts{suffix or "_all"}.csv'
  part.to_csv(csv, index=False)
  print(f'{MODE}: {len(part)} passages in {csv}', flush=True)
  for model in MODELS:
    segs = {'flash': FLASH, 'glm53': GLM53}[model]
    tag = model + suffix
    left = SESSION_LIMIT_S - (time.time() - T0)
    # a model is written only at its very end, so start it only if it can finish
    need = 600
    if model in FORECAST:
        fixed, compute = FORECAST[model]
        need = 1.15 * (fixed + compute * len(part) / len(train))
    if need > left - 300:
        print(f'{tag}: not started, it needs about {need / 3600:.1f} h and {left / 3600:.1f} h are left. '
              'Run it in a session of its own, or as more parts (SHARDS).', flush=True)
        continue
    # f32 state, as in the GPU runs (auto would drop to bf16 when the disk is short)
    cmd = [sys.executable, str(SRC / 'run_pipeline.py'), '--bin', str(BIN), '--segs', json.dumps(segs),
           '--csv', str(csv), '--out', str(OUT), '--tag', tag, '--work', str(WORK / 'state'),
           '--seq-batch', str(SEQ_BATCH), '--state-dtype', 'f32', '--max-text-tokens', str(MAX_TEXT_TOKENS),
           '--prefix', PREFIX, '--threads', str(THREADS)]
    t = time.time()
    # its own process group, so that stopping it also stops ntl-segrun under it, on a timeout or an
    # interrupt alike (otherwise a re-run would race a detached one on the same state file)
    p = subprocess.Popen(cmd, start_new_session=True)
    try:
        code = p.wait(timeout=left - 300)   # stop in time for Kaggle to keep the output
    except BaseException as e:
        os.killpg(p.pid, 9)
        p.wait()
        if not isinstance(e, subprocess.TimeoutExpired):
            raise
        print(f'{tag}: stopped at the session limit after {time.time() - t:.0f} s; nothing written for it. '
              f'Use more parts (SHARDS) or one model per session.', flush=True)
        stop = True
        break
    assert code == 0, f'{tag}: run_pipeline failed ({code}), see {OUT}/{tag}_run.log'
    timing = json.loads((OUT / f'{tag}_timing.json').read_text())
    print(f"{tag}: {timing['n_done']} texts, {timing['n_rows']} token rows, {len(timing['chunks'])} chunk(s), "
          f"{time.time() - t:.0f} s", flush=True)
    if MODE == 'timing':   # same tokens as the GPU timing run: 5,175 rows over the first 40 passages
        assert timing['n_rows'] == 5175, f"{tag}: {timing['n_rows']} token rows, the GPU run had 5175"
"""

TPU_REPORT_CODE = """
if MODE == 'timing':
    # one plan for all of train.csv: each session pays the setup again, then runs its models one
    # after the other; a part of a model (SHARDS) reads every segment again
    n_full = len(pd.read_csv(TRAIN))
    setup_s = json.loads((OUT / 'setup.json').read_text())['setup_s']
    usable = SESSION_LIMIT_S - 300 - setup_s
    cost = {}
    for model in ('flash', 'glm53'):
        if (OUT / f'{model}_timing.json').exists():
            fixed, compute = forecast(model, n_full)
            cost[model] = (1.15 * fixed, 1.15 * compute)   # margin
    sessions = []
    if len(cost) == 2 and sum(f + c for f, c in cost.values()) <= usable:
        sessions.append({'models': ['flash', 'glm53'], 'shard': [0, 1]})
    else:
        for model, (fixed, compute) in cost.items():
            if fixed >= usable:
                print(f'{model}: loading alone takes {fixed / 3600:.1f} h: this machine cannot run it', flush=True)
                continue
            parts = 1
            while fixed + compute / parts > usable:
                parts += 1
            sessions += [{'models': [model], 'shard': [k, parts]} for k in range(parts)]
    total = sum(sum(cost[m][0] + cost[m][1] / s['shard'][1] for m in s['models']) for s in sessions)
    quota_h = (total + len(sessions) * setup_s) / 3600
    print(f'plan: {len(sessions)} full session(s), about {total / 3600:.1f} h of scoring '
          f'plus {setup_s / 3600:.2f} h of setup each, {quota_h:.1f} h of TPU quota in all:', flush=True)
    for s in sessions:
        print(f"  --mode full --models {','.join(s['models'])} --shard {s['shard'][0]}/{s['shard'][1]}", flush=True)
    if quota_h > 20:
        print(f'WARNING: {quota_h:.1f} h is more than the weekly 20 h TPU quota', flush=True)
    (OUT / 'plan.json').write_text(json.dumps({'sessions': sessions, 'forecast': cost}, indent=1))
for f in sorted(OUT.glob('*_features.csv')):
    print(f.name, len(pd.read_csv(f)), 'passages')
sh(f'ls -la {OUT}; du -sh {OUT}', check=False)
print(f'total {time.time() - T0:.0f} s')
"""


def forecast_from(timing_dir: Path, n_texts_full: int) -> dict[str, list[float]]:
    """(fixed s, compute s for all passages) per model, from a downloaded timing run's output: the
    same arithmetic as the notebook's forecast(), so a full-mode session knows what fits."""
    out = {}
    for model in ("flash", "glm53"):
        tj, log = timing_dir / f"{model}_timing.json", timing_dir / f"{model}_run.log"
        if not (tj.exists() and log.exists()):
            continue
        timing = json.loads(tj.read_text())
        done = re.findall(
            r"done\. load ([0-9.]+) s, compute ([0-9.]+) s", log.read_text(errors="replace")
        )
        compute = sum(float(b) for _, b in done)
        rows_full = timing["n_rows"] / timing["n_done"] * n_texts_full
        out[model] = [timing["total_s"] - compute, compute * rows_full / timing["n_rows"]]
    return out


def build_tpu(
    mode: str = "timing",
    models: tuple[str, ...] = ("flash", "glm53"),
    shards: tuple[tuple[int, int], ...] = ((0, 1),),
    forecast: dict[str, list[float]] | None = None,
) -> nbf.NotebookNode:
    assert mode in ("probe", "timing", "full")
    assert set(models) <= {"flash", "glm53"} and models, models
    assert shards and all(0 <= k < n for k, n in shards), shards
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    payload = segrun_payload()
    cells = [
        md(
            "# Name That LLM: GLM-5.3 likelihood features for train.csv, on a TPU VM's CPUs\n\n"
            "The same scoring as the GPU notebook (24 likelihood features per passage under GLM-5.3 and "
            "GLM-5.3-Flash, written to `glm_ll/`), for the TPU VM v5e-8 accelerator. llama.cpp cannot "
            "use the TPU chips, so this notebook builds `ntl-segrun` for the VM's host CPUs and runs it "
            "there with one thread per core. Needs **internet on** (to clone llama.cpp).\n\n"
            "`MODE`: `probe` checks the machine and the inputs only; `timing` scores the first "
            f"{TIMING_ROWS} train passages (the same ones as the GPU timing run) and forecasts the full "
            "run; `full` scores all of train.csv, or parts of it in order with "
            "`SHARDS = [(k, n), ...]` (part k of n: passages k, k+n, k+2n, ...). Each part is saved as "
            "soon as it finishes, so a session that ends early keeps the parts it completed. `MODELS` "
            "picks the models of this session.\n\n"
            "**Caveat.** Kaggle staff have said that a TPU session whose code uses only the CPUs for "
            "about 30 minutes is stopped (Kaggle discussion 607202, 2025-10-28). This notebook never "
            "touches the TPU, so expect that stop; it does not try to get around it.\n\n"
            "The scoring pipeline (`ntl-segrun` and its llama.cpp patch, `validate.sh`, "
            "`run_pipeline.py`, `features.py`, `segio.py`) is José Padilla's, from "
            "*ntl-glm-likelihood-train* and *0.689 | Real GLM-5.3 Likelihood on 2xT4*; the segment "
            "GGUFs are jamesliebel's mirrors of jinghengwu's quantization of Z.ai's open weights. "
            "This notebook only builds them for a CPU and runs them over train.csv."
        ),
        code(
            f"MODE = {mode!r}  # 'probe' | 'timing' | 'full'\n"
            f"MODELS = {tuple(models)!r}  # the models scored in this session, in this order\n"
            f"SHARDS = {[tuple(x) for x in shards]!r}  # full mode: parts in order; merge them offline\n"
            "# from the timing run (builder --forecast): model -> (fixed s, compute s for all of train.csv)\n"
            f"FORECAST = {json.dumps(forecast or {})}\n"
            "THREADS = 0  # 0: one per physical core\n"
            "SEQ_BATCH = 8  # texts per llama_decode, as in the GPU runs\n"
            "SESSION_LIMIT_S = 8.5 * 3600  # stop starting work after this (TPU sessions end at 9 h)\n"
            f"PREFIX = {PREFIX!r}\n"
            f"MAX_TEXT_TOKENS = {MAX_TEXT_TOKENS}\n"
            f"TIMING_ROWS = {TIMING_ROWS}\n"
            f"LLAMA_TAG = {LLAMA_TAG!r}\n"
            f"SEGMENTS = {json.dumps(SEGMENTS)}\n"
            f"SEGRUN_SHA256 = {json.dumps(SEGRUN_FILES, indent=1)}\n" + TPU_SETUP_CODE
        ),
        code("SEGRUN_B64 = " + json.dumps(payload, indent=1) + "\n"),
    ]
    # our pipeline files go next to the author's validation scripts (validate.sh imports segio from there)
    for name in ("segio", "features", "run_pipeline"):
        cells.append(code(f"%%writefile /tmp/ntl/src/segrun/{name}.py\n" + pipeline_source(name)))
    cells += [
        code("if MODE != 'probe':\n" + indent(TPU_BUILD_CODE)),
        code("if MODE != 'probe':\n" + indent(TPU_RUN_CODE)),
        code(FORECAST_CODE.strip() + "\n\n\n" + TPU_REPORT_CODE.strip() + "\n"),
    ]
    return finish(cells)


PUSH_DIR_OPT = typer.Option(None, help="Also write the notebook and kernel-metadata.json here.")
FORECAST_OPT = typer.Option(
    None, help="TPU: folder with a downloaded timing run's output, to size each full session."
)
PATH_OPT = typer.Option(
    None, help="Notebook path (default: notebooks/kaggle_glm_likelihood_train[_tpu].ipynb)."
)


@app.command()
def main(
    mode: str = typer.Option(
        "full", help="'timing' (first 40 passages), 'full', or 'probe' (TPU only)."
    ),
    tpu: bool = typer.Option(
        False, help="The TPU VM variant: builds ntl-segrun for the host CPUs."
    ),
    models: str = typer.Option("flash,glm53", help="TPU: the models of this session, in order."),
    shard: str = typer.Option(
        "0/1", help="TPU full mode: parts k/n to run in order, comma-separated (e.g. 0/4,1/4)."
    ),
    forecast: Path = FORECAST_OPT,
    path: Path = PATH_OPT,
    push_dir: Path = PUSH_DIR_OPT,
    user: str = typer.Option("", help="Kaggle user name, for --push-dir."),
    slug: str = typer.Option(
        "", help="Kaggle notebook slug (default: ntl-glm-likelihood-train-features[-tpu])."
    ),
):
    path = path or (TPU_NOTEBOOK_PATH if tpu else NOTEBOOK_PATH)
    slug = slug or ("ntl-glm-likelihood-train-features" + ("-tpu" if tpu else ""))
    if tpu:
        shards = tuple(tuple(int(x) for x in s.split("/")) for s in shard.split(","))
        fc = (
            forecast_from(forecast, len(pd.read_csv(RAW_DATA_DIR / "train.csv")))
            if forecast
            else None
        )
        if mode == "full" and not fc:
            typer.echo(
                "warning: no --forecast: the session cannot tell in advance whether a model fits"
            )
        nb = build_tpu(mode, tuple(models.split(",")), shards, fc)
    else:
        nb = build(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    typer.echo(f"wrote {path} (MODE={mode}, tpu={tpu})")
    if push_dir is not None:
        assert user, "--push-dir needs --user"
        push_dir.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, push_dir / path.name)
        meta = {
            "id": f"{user}/{slug}",
            # Kaggle derives a new notebook's slug from its title, so the title must spell the slug
            "title": " ".join(
                w.upper() if w in ("ntl", "glm", "tpu") else w for w in slug.split("-")
            ),
            "code_file": path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": not tpu,
            "enable_tpu": tpu,
            "enable_internet": tpu,
            "machine_shape": TPU_MACHINE_SHAPE if tpu else "NvidiaTeslaT4",
            "dataset_sources": SEGMENT_DATASETS,
            "kernel_sources": [] if tpu else [BINARY_NOTEBOOK],
            "competition_sources": ["name-that-llm"],
            "model_sources": [],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        typer.echo(f"wrote {push_dir / 'kernel-metadata.json'}")


if __name__ == "__main__":
    app()
