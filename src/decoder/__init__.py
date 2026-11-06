"""SSVEP decoder proof of concept. CPU only.

Importing this package hides any GPU (CUDA_VISIBLE_DEVICES="") and pins BLAS to 1 thread
per process, because the experiments parallelise over subjects with joblib instead.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Use all cores but one, so the machine stays usable.
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
SEED = 0
