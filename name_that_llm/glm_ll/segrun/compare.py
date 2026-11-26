"""Compare the scores of the segmented runs against the whole-model run (see validate.sh)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import segio  # noqa: E402


def main(d):
    ref = segio.read_scores(os.path.join(d, 'scores_full_b1.bin'))
    R = np.concatenate(ref)
    print(f'whole model: {len(ref)} texts, {len(R)} scored tokens, '
          f'mean logprob {R[:, 0].mean():.4f}, mean entropy {R[:, 1].mean():.4f}, mean rank {R[:, 3].mean():.2f}')
    ok = True
    for p in sorted(glob.glob(os.path.join(d, 'scores_*.bin'))):
        name = os.path.basename(p)[len('scores_'):-4]
        if name == 'full_b1':
            continue
        S = segio.read_scores(p)
        assert [len(x) for x in S] == [len(x) for x in ref], name
        S = np.concatenate(S)
        dlp = np.abs(S[:, 0] - R[:, 0]).max()
        dh = np.abs(S[:, 1] - R[:, 1]).max()
        dmean = abs(S[:, 0].mean() - R[:, 0].mean())
        nrank = int((S[:, 3] != R[:, 3]).sum())
        # f32 states must match to float noise; f16/bf16 states round the residual stream between segments
        tol = 1e-3 if ('f32' in name or name.startswith('full')) else 5e-2
        good = dlp < tol and dh < tol
        ok &= good
        print(f'{name:16s} max|d logprob| {dlp:.2e}  max|d entropy| {dh:.2e}  |d mean logprob| {dmean:.2e}  '
              f'rank changes {nrank}/{len(R)}  {"OK" if good else "FAIL"}')
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(sys.argv[1])
