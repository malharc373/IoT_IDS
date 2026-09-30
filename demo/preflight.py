#!/usr/bin/env python3
"""Verify that the IoT-IDS demo can run on the current machine.

Examples:
    python demo/preflight.py
    python demo/preflight.py --runtime-only
    sudo python demo/preflight.py --runtime-only --iface eth0

Warnings describe optional or generated inputs. Any blocking failure produces
an actionable repair command and a non-zero exit status.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAILED: list[tuple[str, str]] = []
WARNED: list[tuple[str, str]] = []


def _ok(name: str, detail: str = "") -> None:
    print(f"  OK    {name}{' — ' + detail if detail else ''}")


def _warn(name: str, detail: str, fix: str = "") -> None:
    WARNED.append((name, fix))
    print(f"  WARN  {name} — {detail}")
    if fix:
        print(f"        fix: {fix}")


def _fail(name: str, detail: str, fix: str = "") -> None:
    FAILED.append((name, fix))
    print(f"  FAIL  {name} — {detail}")
    if fix:
        print(f"        fix: {fix}")


def _section(title: str) -> None:
    print(f"\n{title}")


def check_python() -> None:
    _section("1. Interpreter")
    version = sys.version_info
    if version < (3, 10):
        _fail(
            "Python version",
            f"{version.major}.{version.minor} is too old",
            "install Python 3.10 or newer",
        )
    else:
        _ok("Python version", f"{version.major}.{version.minor}.{version.micro}")


RUNTIME_DEPS = (
    ("numpy", "pip install -r deploy/requirements-pi.txt"),
    ("onnxruntime", "pip install -r deploy/requirements-pi.txt"),
    ("scapy", "pip install -r deploy/requirements-pi.txt"),
)
DEVELOPMENT_DEPS = ("sklearn", "xgboost", "pandas", "matplotlib")


def check_dependencies(runtime_only: bool) -> None:
    _section("2. Dependencies")
    for module_name, fix in RUNTIME_DEPS:
        try:
            module = importlib.import_module(module_name)
            _ok(module_name, str(getattr(module, "__version__", "installed")))
        except Exception as exc:
            _fail(module_name, repr(exc), fix)

    if runtime_only:
        return
    for module_name in DEVELOPMENT_DEPS:
        try:
            module = importlib.import_module(module_name)
            _ok(module_name, str(getattr(module, "__version__", "installed")))
        except Exception:
            _warn(
                module_name,
                "development dependency is not installed",
                "pip install -r requirements.txt",
            )


def check_artifacts() -> None:
    _section("3. Model artifacts")
    artifacts = (
        ("models/live_ids.onnx", True, "edge inference model"),
        ("models/live_meta.json", True, "model and feature contract"),
        ("models/live_ids.h", False, "microcontroller C export"),
    )
    for relative, required, description in artifacts:
        path = os.path.join(ROOT, relative)
        if os.path.isfile(path):
            _ok(relative, f"{os.path.getsize(path) / 1024:.1f} KiB; {description}")
        elif required:
            _fail(relative, "missing", "python src/train_live_model.py")
        else:
            _warn(relative, "missing", "python src/export_c.py --verify")


def check_pipeline() -> None:
    _section("4. Detection pipeline")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from flow_features import FEATURE_CONTRACT_VERSION, FEATURE_NAMES
        from ids_daemon import Detector
    except Exception as exc:
        _fail("live modules", repr(exc))
        return

    metadata_path = os.path.join(ROOT, "models", "live_meta.json")
    if not os.path.isfile(metadata_path):
        return
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        _fail("model metadata", repr(exc), "python src/train_live_model.py")
        return

    if metadata.get("features") != FEATURE_NAMES:
        _fail("feature order", "metadata differs from the live extractor")
        return
    if metadata.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
        _fail("feature semantics", "metadata contract version is stale")
        return
    _ok("feature contract", f"v{FEATURE_CONTRACT_VERSION}; {len(FEATURE_NAMES)} features")

    try:
        detector = Detector()
        verdict = detector.classify([[0.0] * len(FEATURE_NAMES)])
    except (Exception, SystemExit) as exc:
        _fail("ONNX round-trip", str(exc), "python src/train_live_model.py")
        return
    if len(verdict) != 1:
        _fail("ONNX round-trip", f"expected one verdict, received {len(verdict)}")
        return
    label, confidence = verdict[0]
    _ok("ONNX round-trip", f"{label}, confidence={confidence:.3f}")


def check_generated_paths() -> None:
    _section("5. Generated paths")
    generated = (
        ("data/pcaps", "bash demo/run_demo.sh"),
        ("logs", "created on the first daemon run"),
    )
    for relative, fix in generated:
        path = os.path.join(ROOT, relative)
        if os.path.isdir(path) and os.listdir(path):
            _ok(relative, f"{len(os.listdir(path))} item(s)")
        else:
            _warn(relative, "not generated yet", fix)


def check_interface(interface: str) -> None:
    _section("6. Live capture")
    if not sys.platform.startswith("linux"):
        _warn("platform", "interface validation is available on Linux only")
        return
    path = os.path.join("/sys/class/net", interface)
    if os.path.exists(path):
        _ok("interface", interface)
    else:
        try:
            choices = ", ".join(sorted(os.listdir("/sys/class/net")))
        except OSError:
            choices = "run: ip -brief address"
        _fail("interface", f"{interface!r} not found", f"choose one of: {choices}")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        _warn("capture privileges", "not running as root", "rerun with sudo")
    else:
        _ok("capture privileges", "root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="check only dependencies needed by the deployed sensor",
    )
    parser.add_argument("--iface", help="also validate a Linux capture interface")
    args = parser.parse_args()

    print(f"IoT-IDS preflight — {ROOT}")
    check_python()
    check_dependencies(args.runtime_only)
    check_artifacts()
    check_pipeline()
    check_generated_paths()
    if args.iface:
        check_interface(args.iface)

    print("\n" + "=" * 62)
    if FAILED:
        print(f"NOT READY — {len(FAILED)} failure(s), {len(WARNED)} warning(s)")
        return 1
    print(f"READY — {len(WARNED)} non-blocking warning(s)")
    print("next: bash demo/run_demo.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
