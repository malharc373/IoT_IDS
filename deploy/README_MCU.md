# Running the model on a microcontroller (ESP32-class)

For devices too small for a Python/ONNX runtime, the model compiles to a single
dependency-free C header, `models/live_ids.h`:

```bash
python src/export_c.py --verify   # regenerate + check 100% parity vs XGBoost
```

Footprint: ~43 KB of `const` tree data in flash, ~130 bytes of RAM at inference,
no libc math required. Fits comfortably on an ESP32 (4 MB flash / 520 KB RAM);
too large for tiny AVR Arduinos (train a smaller model for those — fewer
estimators / lower depth in `src/train_live_model.py`).

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

## Notes

- The feature extraction (`src/flow_features.py`) is the reference for how to
  compute the 22 features from packets; port the parts you need to C for your
  platform, or run a lightweight flow table on-device.
- Regenerate `live_ids.h` whenever you retrain — it is derived from the exact
  booster and verified byte-for-decision against it.
