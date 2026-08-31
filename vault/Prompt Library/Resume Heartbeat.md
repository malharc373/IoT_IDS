# Prompt — non-duplicating resume heartbeat

```text
Inspect the active TASK file, latest handoff, worker/process/run identity, and
current capacity. If the worker is still active, do not relaunch it. If capacity
has reset and work stopped, resume exactly the recorded next action within the
same scope and exclusions. Update the vault before waiting again. Never broaden
authority or bypass an external/destructive action gate.
```
