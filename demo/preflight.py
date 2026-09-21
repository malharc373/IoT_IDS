#!/usr/bin/env python3
"""
preflight.py — demo-day readiness check for the IoT-IDS sensor.

Answers one question before you stand in front of an audience (or an
examiner): *will the demo run on this machine right now?*  It verifies the
interpreter, the dependencies, the shipped model artifacts, and a real
classification round-trip through the live pipeline, then prints what is
generated on first run rather than shipped in git.

    python demo/preflight.py                 # full dev machine check
    python demo/preflight.py --runtime-only  # Pi: inference deps only
    python demo/preflight.py --iface eth0    # also check live-capture readiness

Exit code 0 = ready. Exit code 1 = something is genuinely broken; the failing
check prints the command that fixes it. Warnings never fail the run — they mark
things that are generated on demand (data/pcaps, logs/) or only needed for
retraining and plots.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GRN, RED, YEL, DIM, BLD, NC = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")

FAILED: list[tuple[str, str]] = []
WARNED: list[tuple[str, str]] = []


def ok(name, detail=""):
    print(f"  {GRN}OK  {NC}  {name}{DIM}{'  ' + detail if detail else ''}{NC}")


def warn(name, detail, fix=""):
    WARNED.append((name, fix))
    print(f"  {YEL}WARN{NC}  {name}  {DIM}{detail}{NC}")
    if fix:
        print(f"        {DIM}→ {fix}{NC}")


def fail(name, detail, fix=""):
    FAILED.append((name, fix))
    print(f"  {RED}FAIL{NC}  {name}  {detail}")
    if fix:
        print(f"        {DIM}→ {fix}{NC}")


def section(title):
    print(f"\n{BLD}{title}{NC}")


# ── 1. interpreter ────────────────────────────────────────────────────────────
def check_python():
    section("1. Interpreter")
    v = sys.version_info
    if v < (3, 9):
        fail("python version", f"{v.major}.{v.minor} is too old",
             "the project is verified on Python 3.10+")
    else:
        ok("python version", f"{v.major}.{v.minor}.{v.micro}")


# ── 2. dependencies ───────────────────────────────────────────────────────────
RUNTIME_DEPS = [
    ("numpy", "pip install -r deploy/requirements-pi.txt"),
    ("onnxruntime", "pip install -r deploy/requirements-pi.txt"),
    ("scapy", "pip install -r deploy/requirements-pi.txt"),
]
TRAINING_DEPS = [
    "sklearn", "xgboost", "pandas", "matplotlib",
]


def check_deps(runtime_only):
    section("2. Dependencies")
    for mod, fix in RUNTIME_DEPS:
        try:
            m = importlib.import_module(mod)
            ok(f"{mod}", getattr(m, "__version__", ""))
        except Exception as e:
            fail(f"{mod} (required for inference)", repr(e), fix)

    if runtime_only:
        print(f"  {DIM}(skipping the training stack — --runtime-only){NC}")
        return

    for mod in TRAINING_DEPS:
        try:
            m = importlib.import_module(mod)
            ok(f"{mod}", getattr(m, "__version__", ""))
        except Exception:
            warn(f"{mod} (training/plots only)", "not installed",
                 "pip install -r requirements.txt  — not needed to run the demo")


# ── 3. shipped model artifacts ────────────────────────────────────────────────
ARTIFACTS = [
    ("models/live_ids.onnx", True, "the edge model (scaler baked in)"),
    ("models/live_meta.json", True, "feature order, labels, categories, metrics"),
    ("models/live_ids.h", False, "C model for microcontrollers"),
]


def check_artifacts():
    section("3. Model artifacts")
    for rel, required, what in ARTIFACTS:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            ok(rel, f"{kb:.1f} KB — {what}")
        elif required:
            fail(rel, "missing", "python src/train_live_model.py")
        else:
            warn(rel, "missing", "python src/export_c.py --verify")


# ── 4. the model actually loads and classifies ────────────────────────────────
def check_pipeline():
    section("4. Detection pipeline")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from flow_features import FEATURE_NAMES
    except Exception as e:
        fail("import flow_features", repr(e))
        return
    ok("import flow_features", f"{len(FEATURE_NAMES)} features")

    meta_path = os.path.join(ROOT, "models", "live_meta.json")
    if not os.path.exists(meta_path):
        return  # already reported in section 3
    meta = json.load(open(meta_path))

    if meta["features"] != list(FEATURE_NAMES):
        fail("train/serve feature parity",
             "models/live_meta.json feature order != flow_features.FEATURE_NAMES",
             "retrain so the extractor and the model agree: "
             "python src/train_live_model.py")
        return
    ok("train/serve feature parity", "extractor order == model order")

    try:
        from ids_daemon import Detector
        det = Detector()
    except SystemExit as e:
        fail("load Detector", str(e))
        return
    except Exception as e:
        fail("load Detector", repr(e), "python src/train_live_model.py")
        return
    ok("load Detector", f"{len(det.labels)} classes, "
                        f"{len(set(det.categories.values()))} categories")

    # a real round-trip through ONNX: a neutral vector must come back labelled
    try:
        verdict = det.classify([[0.0] * len(FEATURE_NAMES)])
        kind, conf = verdict[0]
        ok("classify round-trip", f"-> {kind} (conf {conf:.2f})")
    except Exception as e:
        fail("classify round-trip", repr(e))

    m = meta.get("metrics", {})
    if m:
        ok("recorded in-domain metrics",
           " ".join(f"{k}={v:.3f}" for k, v in m.items()))


# ── 5. generated (not shipped) paths ──────────────────────────────────────────
GENERATED = [
    ("data/pcaps", "synthetic captures", "bash demo/run_demo.sh  (step 1 builds them)"),
    ("logs", "alert feed the dashboard reads", "written on the first daemon run"),
]


def check_generated():
    section("5. Generated paths (not in git)")
    for rel, what, how in GENERATED:
        path = os.path.join(ROOT, rel)
        if os.path.isdir(path) and os.listdir(path):
            n = len(os.listdir(path))
            ok(rel, f"{n} file(s) — {what}")
        else:
            warn(rel, f"absent — {what}", how)


# ── 6. live capture readiness (opt-in) ────────────────────────────────────────
def check_iface(iface):
    section("6. Live capture")
    if os.name != "posix":
        warn("platform", "live capture checked on POSIX only")
        return
    sysnet = f"/sys/class/net/{iface}"
    if os.path.exists(sysnet):
        ok(f"interface {iface}", "present")
    else:
        try:
            names = sorted(os.listdir("/sys/class/net"))
        except OSError:
            names = []
        fail(f"interface {iface}", "not found",
             f"pick one of: {', '.join(names) or 'run: ip -brief addr'}")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        warn("privileges", "not root",
             f"live sniffing needs root: sudo python src/ids_daemon.py --iface {iface}")
    else:
        ok("privileges", "root — can sniff")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-only", action="store_true",
                    help="check inference deps only (the Raspberry Pi runtime set)")
    ap.add_argument("--iface", default=None,
                    help="also check live-capture readiness on this interface")
    args = ap.parse_args()

    print(f"{BLD}IoT-IDS preflight{NC}  {DIM}repo={ROOT}{NC}")
    check_python()
    check_deps(args.runtime_only)
    check_artifacts()
    check_pipeline()
    check_generated()
    if args.iface:
        check_iface(args.iface)

    print(f"\n{'=' * 62}")
    if FAILED:
        print(f"  {RED}NOT READY{NC} — {len(FAILED)} blocking issue(s), "
              f"{len(WARNED)} warning(s)")
        for name, fix in FAILED:
            print(f"    {RED}·{NC} {name}" + (f"  → {fix}" if fix else ""))
        print("=" * 62)
        sys.exit(1)

    print(f"  {GRN}READY{NC} — demo can run"
          + (f"  ({len(WARNED)} warning(s), none blocking)" if WARNED else ""))
    print(f"{DIM}  next: bash demo/run_demo.sh"
          f"   then: python src/dashboard.py{NC}")
    print("=" * 62)


if __name__ == "__main__":
    main()
