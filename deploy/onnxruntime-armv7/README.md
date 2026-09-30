# onnxruntime for 32-bit Raspberry Pi (armv7l)

onnxruntime publishes no wheels for 32-bit ARM, so a Raspberry Pi 2, or any Pi
on a 32-bit OS, cannot `pip install` it. The daemon does not need it there: it
falls back to the C export of the same model (`src/c_backend.py`). This recipe
builds onnxruntime from Microsoft's official source for boards where you want
the ONNX engine anyway, for example to compare engines.

The wheel is **not** committed (11.6 MB binary). Build it with this recipe, or
keep your own copy with its hash.

## Build (on any machine with Docker and arm/v7 emulation)

```bash
cd deploy/onnxruntime-armv7
docker build --platform linux/arm/v7 -t ort-armv7-build .
docker volume create ort-armv7
docker run --platform linux/arm/v7 --name ort-armv7 \
    -v ort-armv7:/work/vol -e JOBS=6 ort-armv7-build
docker cp ort-armv7:/work/vol/out/. ./dist/
```

- Source: `github.com/microsoft/onnxruntime` at tag `v1.23.2` (`ORT_TAG`).
- Target: Cortex-A7, NEON-VFPv4, hard float (`-mcpu=cortex-a7`).
- Base image: Debian trixie, matching Raspberry Pi OS trixie (glibc 2.41,
  Python 3.13, numpy 2.2). The wheel needs glibc 2.38 or newer.
- On an Apple Silicon Mac under QEMU the full build took about 3 hours
  (1,059 build steps, `JOBS=6`).

Built 30 Sep 2026: `onnxruntime-1.23.2-cp313-cp313-linux_armv7l.whl`,
SHA-256 `142db85fb00e1b2d184e89d90f986b3574ca29421f4557515a7daa4c57c003fd`.

## Install on the Pi

Keep it out of the sensor's own venv unless you mean to switch engines:

```bash
python3 -m venv --system-site-packages /tmp/ort-test/venv
/tmp/ort-test/venv/bin/pip install ./onnxruntime-1.23.2-cp313-cp313-linux_armv7l.whl
```

With onnxruntime importable, `ids_daemon.py --backend auto` uses it; pass
`--backend c` to keep the C engine. The warning
`GPU device discovery failed ... /sys/class/drm/card0/device/vendor` is
harmless on a Pi.

## Results on a Raspberry Pi 2 Model B

Same model, 5,000 inputs (see
[`reports/pi2-acceptance-2026-09-30`](../../reports/pi2-acceptance-2026-09-30/README.md)):

- identical labels to onnxruntime on the Mac, max probability difference
  4.8 × 10⁻⁷
- identical labels to the Pi's C engine, max difference 1.4 × 10⁻⁶

Measured back to back on an idle board:

| | onnxruntime 1.23.2 | C engine |
|---|---|---|
| single flow | 331 µs (p99 418 µs) | 511 µs (p99 634 µs) |
| batch of 1,024 | 7,881 flows/s | 17,492 flows/s |

onnxruntime wins on single flows, the C engine on batches. Under attack the
daemon scores large batches, so the C engine stays the better default on this
board.
