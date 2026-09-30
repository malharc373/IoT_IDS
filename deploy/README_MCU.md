# Running the model on a microcontroller (ESP32-class)

> **Status (30 September 2026): inference kernel only.** The model compiles to
> C and runs, but there is no ESP32 intrusion detector yet. What exists and
> what does not:
>
> | Piece | State |
> |---|---|
> | C export of the model, decision parity with XGBoost | done (`src/export_c.py --verify`) |
> | Same C code on real ARM hardware | done: it is the default inference engine on the 32-bit Raspberry Pi 2 |
> | Arduino example sketch (`esp32_iot_ids/`) | written; **not yet compiled with the ESP32 toolchain or flashed** |
> | On-device flow table and 22-feature extraction in C | **not started**; the sketch feeds zeros |
> | Packet capture on the ESP32 | **not started**; see *Placement* below |
> | Accuracy, latency or memory measured on an ESP32 | **none** |
>
> On-device detection is targeted for the final project review (October 2026).

For devices too small for a Python/ONNX runtime, the model compiles to a single
dependency-free C header, `models/live_ids.h`:

```bash
python src/export_c.py --verify   # regenerate + check 100% parity vs XGBoost
```

Footprint, measured from an `-Os` build on the development host: about 49 KB
of `const` data (a 43 KB node table of 2,780 nodes plus a 6 KB index for the
1,200 trees), about 240 B of code, an estimated stack of about 130 bytes, no
heap, and no libc math. The type sizes are the same on an ESP32, so the constant data should
be the same size there, but nothing has been built for the chip yet. That fits
an ESP32 (4 MB flash / 520 KB RAM) with room to spare. It is too large for small
AVR Arduinos; train a smaller model for those (fewer estimators or lower depth
in `src/train_live_model.py`).

## Usage

```c
#include "live_ids.h"

/* Fill the 22 flow features in the exact order of IDS_LABELS' feature set
 * (see models/live_meta.json "features"). Compute them on-device from the
 * packets you observe, then: */
float feats[IDS_NUM_FEATURES] = { /* proto, duration, tot_pkts, ... dst_port */ };
float margin = 0.0f;
int cls = ids_predict_with_margin(feats, &margin); /* class id + score margin */
const char *name = IDS_LABELS[cls];      /* "benign", "portscan", ... */
if (cls != 0) {
    /* attack detected — raise a GPIO, publish an MQTT alert, drop the peer … */
}
```

There is no scaler — pass **raw** feature values. `ids_predict_with_margin`
returns the arg-max class and writes the gap between the best and second-best
raw class scores. A larger margin means the model's choice was less ambiguous,
but it is **not a calibrated probability** and must not be interpreted as one.
The original `ids_predict` class-only function remains available. Both are pure
C99 and use no heap allocation, so they are safe in an ISR-adjacent loop.

## ESP32 serial example

An Arduino/PlatformIO example is provided at
`deploy/esp32_iot_ids/esp32_iot_ids.ino`. Copy or symlink `models/live_ids.h`
into that sketch directory, replace the sample feature acquisition function
with your flow counter, and flash it. The example prints the predicted class
and raw score margin over Serial without enabling any enforcement action.

As shipped it passes an all-zero feature vector, which the model classifies as
`icmpflood`. That output only proves the call path works; it is not a
detection. The sketch has not been compiled for the ESP32 yet.

## Placement: what an ESP32 can actually see

The model was trained on bidirectional IP flows. An ESP32 can only compute
those flows for traffic that passes through it:

- **Inline (recommended):** the ESP32 is the Wi-Fi access point or a bridge the
  IoT devices use, so their IP traffic crosses it. It can then build the same
  flows the model expects.
- **Passive Wi-Fi sniffing:** promiscuous mode sees 802.11 frames on one channel
  and cannot see a switched wired LAN. The features would differ from the
  training data, so this needs a new feature set and a retrained model.

## Remaining work for an ESP32 detector

1. Choose the placement above (inline keeps the current model valid).
2. Port the flow table and the 22 features of `src/flow_features.py` to C, and
   test it against the Python version on the same capture.
3. Add the capture path (ESP-IDF netif hook for inline, or the Wi-Fi
   promiscuous callback).
4. Feed real features to `ids_predict_with_margin`, and report attacks over
   Serial, GPIO or MQTT.
5. Measure on the board: agreement with the Pi and host, latency, RAM, flows
   per second.
6. If memory or latency is tight, train and export a smaller model.

## Notes

- The feature extraction (`src/flow_features.py`) is the reference for how to
  compute the 22 features from packets; port the parts you need to C for your
  platform, or run a lightweight flow table on-device.
- Regenerate `live_ids.h` whenever you retrain — it is derived from the exact
  booster and verified byte-for-decision against it.
