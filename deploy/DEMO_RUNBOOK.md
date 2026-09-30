# Live demo runbook — IoT-IDS on the Raspberry Pi

One page for driving the on-device demonstration. Everything below runs **on the
Pi** except where it says "laptop". All attack traffic stays on the Pi (floods
target loopback), so the demo is self-contained and does not touch the venue
network.

> Do **not** enable IPS enforce mode on the venue network. On real traffic this
> model raises false alarms that would throttle/block innocent devices,
> including the gateway. Keep the sensor in detection (or dry-run IPS) for the
> talk. The IPS story is told as a *result*, not run live.

## 0. Before the audience (5 min)

```bash
# on the Pi
cd ~/IOT-IDS
git fetch origin && git checkout main && git pull        # after PR #9 is merged
# (if #9 is not merged yet: git checkout fix/ips-target-guard && git pull)
ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}'     # note the Pi's IP
```

Confirm the services are up and healthy:

```bash
systemctl is-active iot-ids iot-ids-dashboard
tail -n 2 logs/sensor_status.json          # should show a fresh timestamp
```

Open the dashboard. On the Pi's own screen use Firefox (Chromium renders blank
on the Pi 2):

```bash
firefox --kiosk http://127.0.0.1:8080 &
```

From the laptop instead, tunnel and open locally (no token needed):

```bash
# laptop
ssh -L 8080:127.0.0.1:8080 malharfalke@<PI_IP>
# then browse to http://127.0.0.1:8080 on the laptop
```

You should see: sensor **LIVE**, packet/flow counters ticking, an empty or quiet
incident list.

## 1. The demo (about 3–4 min)

Run the guided attack script on the Pi. `sudo` lets the raw-packet floods and
the Xmas scan run:

```bash
sudo python attacks/live_demo.py --only portscan synflood slowloris xmas_scan
```

These four were selected because the earlier controlled Pi run detected them
clearly (portscan ~1.00, xmas_scan ~0.97, synflood ~0.99, slowloris ~0.92).
That evidence does not guarantee identical labels in a fresh loopback run, so
describe what the dashboard actually shows. The script narrates each step and
pauses after it.

Talking points as it runs:
- Benign background traffic runs throughout to give the attack timeline
  context. It can raise false incidents: the measured LAN baseline was about
  1.9 incidents/min, which is why this remains an IDS-first demonstration.
- Each attack shows up as a **per-source incident** with a category/type label
  and a confidence, not a flood of raw packet alerts.
- Classification runs locally on the Pi. State the active backend shown by the
  service logs; both the C engine and the separately built 32-bit ARM
  onnxruntime were benchmarked, so do not imply only one can run on the board.

To show the full range (all nine classes, ~5 min): drop `--only`. Add `--fast`
for a shorter run.

## 2. If you have time — the honest limits (spoken, not run)

- On real LAN traffic the synthetic-trained model raises ~1.9 false-alarm
  incidents/min (mostly IPv6 link-local chatter it never saw in training).
- Because those false alarms would become firewall actions, the IPS ships in
  **dry-run**; a real enforce test blocked innocent devices. This is the same
  message as the ~100% synthetic accuracy: the numbers reflect the generators,
  and real-traffic robustness is the open work.
- Full evidence: `reports/pi2-live-2026-09-30/` and
  `reports/pi2-acceptance-2026-09-30/`.

## 3. Reset between runs

```bash
# Preserve the evidence log; note the current time to separate the next run.
date -Iseconds
sudo systemctl restart iot-ids
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard blank on the Pi screen | Use Firefox kiosk, not Chromium (EGL unsupported on Pi 2). |
| Dashboard shows **SILENT** / stale | `sudo systemctl restart iot-ids`; check `journalctl -u iot-ids -n 30`. |
| `attacks/live_demo.py: No such file` | You are on an old branch; `git pull` (see §0). |
| Floods skipped: "needs root" | Re-run the script with `sudo`. |
| `nmap`/`hping3` not found | `sudo apt install -y nmap hping3`. |
| Venue network flaky / no laptop→Pi route | No problem — the floods target the Pi's own loopback; run the demo entirely on the Pi's screen. |
| Attack fired but no incident | Give it ~2–5 s (the sensor flushes incidents on a timer); watch the counters move. |

## Do not

- Do not pass `--target <someone-else>` — keep the default loopback target.
- Do not add `--ips --prevent` on the venue network.
- Do not run the floods against any host you do not own.
