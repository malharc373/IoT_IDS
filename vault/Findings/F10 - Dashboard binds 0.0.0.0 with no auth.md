---
title: F10 — Dashboard binds 0.0.0.0 with no auth
tags: [finding, significant, dashboard, security, performance]
severity: significant
status: fixed
files: ["src/dashboard.py", "src/ids_daemon.py", "deploy/setup_pi.sh"]
date: 2026-08-19
---

# F10 — Dashboard binds 0.0.0.0 with no auth, and re-parses the log on every poll

Three problems in one component.

## 1. Unauthenticated exposure to the LAN

```python
ap.add_argument("--host", default="0.0.0.0")
```

…and `deploy/setup_pi.sh` enabled the dashboard service at boot. So a fresh Pi
install served, to every host on the segment, with no authentication:

- every attacking source IP and attack type
- every currently blocked host and when the block expires
- the internal addressing of the monitored network
- a live 2-second-refreshing view of what the sensor can and cannot see

That is reconnaissance material handed to exactly the population the sensor
exists to watch — and an attacker who can see their own block expiry can simply
wait it out. An odd thing to ship unauthenticated in a security project.

## 2. Full log re-parse on every request

```python
with open(log_path) as f:
    lines = f.readlines()[-5000:]      # whole file into memory, then slice
```

`build_state()` ran this **per request**, and the page polls every 2 seconds,
per connected client. Combined with an alert log that had no size bound
(problem 3), cost grew without limit: a week-old sensor with a 500 MB feed
would read 500 MB from disk every 2 seconds per viewer.

## 3. No log rotation anywhere

`AlertLog` opened the file in append mode and never bounded it. Nothing in the
repo rotated `logs/alerts.jsonl`.

## The fixes

### Binding and auth

Default bind is now `127.0.0.1`. Binding to a non-loopback address **requires**
a token or an explicit acknowledgement — the process refuses to start otherwise:

```
[ERROR] refusing to serve the alert feed on 0.0.0.0 without auth.
        The dashboard exposes attacking hosts, blocked hosts and the
        addressing of the monitored segment.
        Pick one:
          --token generate      mint a token and print the URL
          --token-file PATH     read a private token file
          --host 127.0.0.1      keep it local, reach it over an SSH tunnel:
                                  ssh -L 8080:127.0.0.1:8080 pi@<host>
          --insecure            you have read the above and accept it
```

Tokens are accepted as `?token=…` or an `X-Auth-Token` header and compared with
`hmac.compare_digest`, so the token cannot be recovered by timing. Responses
carry `Content-Security-Policy` (the page only ever renders its own inlined
CSS/JS and calls itself), `X-Content-Type-Options: nosniff` and
`Referrer-Policy: no-referrer`.

`deploy/setup_pi.sh` now mints a token on first run, stores it at
`logs/dashboard.token` mode 600, reuses it across re-runs so bookmarked URLs
keep working, passes it to the unit via `Environment=`, and prints both the
tokenised URL and the SSH-tunnel alternative.

> [!note] No TLS
> This is a stdlib `http.server`. The token authenticates but does not encrypt.
> Over an untrusted network the SSH tunnel is the right answer, and the
> docstring says so.

### Incremental reads

`AlertFeed` keeps a bounded `deque` and reads only bytes appended since the last
poll, tracking the offset and inode. It detects rotation (inode changed) and
truncation (file shrank) and re-reads cleanly, and holds back a trailing
half-written line until the writer completes it — the daemon appends and flushes
concurrently, so a partial record is a normal occurrence, not an error.

### Rotation

`AlertLog` gained size-based rotation (`--log-max-mb`, default 32, `0` disables)
with `N` numbered backups. The dashboard's inode check means rotation is picked
up without a restart.

## Verification

Three new tests:

- `dashboard auth + binding` — `_is_loopback` classification; the off-loopback
  no-auth start **exits non-zero** with "refusing to serve"; unauthenticated
  requests get 401 while both token transports get 200
- `dashboard incremental reads` — a file that has not grown is not re-read; an
  appended record is picked up; a partial line is held back until complete;
  rotation to a smaller file resets cleanly rather than seeking past the end
- `alert log rotation` — the live log stays under its bound, a `.1` backup
  appears, and no more backups than configured are kept

Full suite: 32 passed, 0 failed.

## Bonus

The dashboard now also renders the IPS **throttle** tier introduced in
[[F08 - Rate limiting is documented but not implemented]], and tolerates the
pre-2026-08 flat `ips_state.json` format.

## Related

[[F08 - Rate limiting is documented but not implemented]] · [[Architecture]]
