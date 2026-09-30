/* Minimal ESP32/Arduino inference harness for the IoT-IDS C model.
 * Copy models/live_ids.h beside this sketch before compiling.
 * This demonstrates inference only; production feature extraction must match
 * models/live_meta.json and src/flow_features.py exactly.
 */
#include <Arduino.h>
#include "live_ids.h"

static void collect_flow_features(float features[IDS_NUM_FEATURES]) {
  /* Replace with a bounded on-device flow table. Zeroes are only a compile/run
   * demonstration and are not a meaningful network observation. */
  for (int i = 0; i < IDS_NUM_FEATURES; ++i) features[i] = 0.0f;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("IoT-IDS ESP32 inference example");
}

void loop() {
  float features[IDS_NUM_FEATURES];
  float margin = 0.0f;
  collect_flow_features(features);
  int cls = ids_predict_with_margin(features, &margin);

  Serial.print("class=");
  Serial.print(IDS_LABELS[cls]);
  Serial.print(" raw_margin=");
  Serial.println(margin, 6);
  delay(5000);
}
