# ntl-segrun sources (third-party)

Copied unchanged from the source archive embedded in José Padilla's public Kaggle notebook
[`estopadilla/ntl-glm-likelihood-train`](https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train)
(archive sha256 `fe1d2833cda498559174e5a05cb8eba04e0c1143a7d98993a29ceb74fb1a1d16`):

- `llama-cpp-b12247-ntl.patch`: the `ntl-segrun` tool and the layer-segment support, applied to llama.cpp tag b12247
- `validate.sh`, `slice_gguf.py`, `compare.py`: the author's check that a model run by layer segments scores the same as the whole model

`kaggle_glm_features.py --tpu` writes them into the TPU notebook, which builds `ntl-segrun` for the CPU.
The files must keep LF line endings (see `.gitattributes`); the notebook builder checks their sha256.
