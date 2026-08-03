# Running the model on a microcontroller (ESP32-class)

For devices too small for a Python/ONNX runtime, the model compiles to a single
dependency-free C header, `models/live_ids.h`:

```bash
python src/export_c.py --verify   # regenerate + check 100% parity vs XGBoost
```

Footprint: ~42 KB of `const` tree data in flash, ~130 bytes of RAM at inference,
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
int cls = ids_predict(feats);            /* class id */
const char *name = IDS_LABELS[cls];      /* "benign", "portscan", ... */
if (cls != 0) {
    /* attack detected — raise a GPIO, publish an MQTT alert, drop the peer … */
}
```

The scaler is baked in — pass **raw** feature values; `ids_predict` scales,
walks every tree, and returns the arg-max class. It is pure C99 and has no
heap allocation, so it is safe to call from an ISR-adjacent loop.

## Notes

- The feature extraction (`src/flow_features.py`) is the reference for how to
  compute the 22 features from packets; port the parts you need to C for your
  platform, or run a lightweight flow table on-device.
- Regenerate `live_ids.h` whenever you retrain — it is derived from the exact
  booster and verified byte-for-decision against it.
