#!/usr/bin/env python3
"""
c_backend.py — native C inference for hosts where onnxruntime is unavailable.

onnxruntime publishes no wheels for 32-bit ARM (armv7l), so a Raspberry Pi 2,
or any Pi running a 32-bit OS, cannot load `models/live_ids.onnx`. The same
tree ensemble is also exported as dependency-free C in `models/live_ids.h`
(src/export_c.py). This module compiles a small shared library around that
header and calls it through ctypes, so `ids_daemon.Detector` can run with only
numpy plus a C compiler.

Output is the same as the ONNX model: the header gives per-class raw scores
(the sum of leaf values), and softmax over them is XGBoost's multi:softprob
probability. The constant base_score shifts every class equally and cancels.
tests/smoke_test.py checks labels and probabilities against onnxruntime.

The library is built on first use and cached under build/c_backend/, keyed by
a hash of the header and wrapper source, so a re-exported header is never
served by a stale binary.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_HEADER = os.path.join(ROOT, "models", "live_ids.h")
DEFAULT_BUILD_DIR = os.path.join(ROOT, "build", "c_backend")

# Exposes the header's static tables through a stable C ABI. The tree walk
# mirrors ids_predict_with_margin() but keeps every class score, because the
# daemon's confidence gates need probabilities, not just the arg-max.
_WRAPPER = r"""
#include "live_ids.h"

int ids_num_features(void) { return IDS_NUM_FEATURES; }
int ids_num_class(void) { return IDS_NUM_CLASS; }
int ids_contract_version(void) { return IDS_FEATURE_CONTRACT_VERSION; }
const char *ids_label(int i) {
    return (i >= 0 && i < IDS_NUM_CLASS) ? IDS_LABELS[i] : 0;
}

void ids_scores_batch(const float *x, int n_rows, float *scores) {
    int r, t, n, i;
    for (r = 0; r < n_rows; r++) {
        const float *row = x + (long)r * IDS_NUM_FEATURES;
        float *out = scores + (long)r * IDS_NUM_CLASS;
        for (i = 0; i < IDS_NUM_CLASS; i++) out[i] = 0.0f;
        for (t = 0; t < IDS_NUM_TREES; t++) {
            n = IDS_TREE_ROOT[t];
            while (IDS_NODES[n].feature >= 0)
                n = (row[IDS_NODES[n].feature] < IDS_NODES[n].value)
                    ? IDS_NODES[n].yes : IDS_NODES[n].no;
            out[IDS_TREE_CLASS[t]] += IDS_NODES[n].value;
        }
    }
}

/* Keep the class-only API referenced so -Wunused-function stays quiet. */
int ids_predict_one(const float *x) { return ids_predict(x); }
"""


def _compiler():
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")


def build_library(header_path=DEFAULT_HEADER, build_dir=DEFAULT_BUILD_DIR):
    """Compile (or reuse) the shared library for `header_path`; return its path."""
    if not os.path.isfile(header_path):
        raise FileNotFoundError(
            f"C model header not found: {header_path}\n"
            f"        Run: python src/export_c.py --verify")
    with open(header_path, "rb") as fh:
        header = fh.read()
    digest = hashlib.sha256(header + _WRAPPER.encode()).hexdigest()[:16]
    lib_path = os.path.join(build_dir, f"live_ids_{digest}.so")
    if os.path.isfile(lib_path):
        return lib_path

    cc = _compiler()
    if not cc:
        raise RuntimeError(
            "onnxruntime is unavailable and no C compiler was found to build "
            "the native fallback; install gcc (apt-get install gcc)")
    os.makedirs(build_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="idsc_", dir=build_dir)
    try:
        shutil.copy(header_path, os.path.join(work, "live_ids.h"))
        src = os.path.join(work, "wrapper.c")
        with open(src, "w") as fh:
            fh.write(_WRAPPER)
        tmp_lib = os.path.join(work, "lib.so")
        subprocess.run([cc, "-std=c99", "-O2", "-shared", "-fPIC",
                        "-I", work, "-o", tmp_lib, src],
                       check=True, capture_output=True, text=True)
        os.replace(tmp_lib, lib_path)   # atomic: concurrent builders are safe
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"C backend build failed:\n{exc.stderr}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return lib_path


class CModel:
    """ctypes wrapper with the same (labels, probabilities) contract as ONNX."""

    def __init__(self, header_path=DEFAULT_HEADER, build_dir=DEFAULT_BUILD_DIR):
        self.lib_path = build_library(header_path, build_dir)
        lib = ctypes.CDLL(self.lib_path)
        for name in ("ids_num_features", "ids_num_class", "ids_contract_version"):
            getattr(lib, name).restype = ctypes.c_int
        lib.ids_label.restype = ctypes.c_char_p
        lib.ids_label.argtypes = [ctypes.c_int]
        lib.ids_scores_batch.restype = None
        lib.ids_scores_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)]
        self._lib = lib
        self.n_features = lib.ids_num_features()
        self.num_class = lib.ids_num_class()
        self.contract_version = lib.ids_contract_version()
        self.labels = [lib.ids_label(i).decode() for i in range(self.num_class)]

    def scores(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"expected shape (n, {self.n_features}), got {X.shape}")
        out = np.empty((X.shape[0], self.num_class), dtype=np.float32)
        fptr = ctypes.POINTER(ctypes.c_float)
        self._lib.ids_scores_batch(X.ctypes.data_as(fptr), X.shape[0],
                                   out.ctypes.data_as(fptr))
        return out

    def predict(self, X):
        """Return (labels, probabilities) like the ONNX classifier outputs."""
        s = self.scores(X).astype(np.float64)
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        return probs.argmax(axis=1).astype(np.int64), probs
