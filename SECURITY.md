# Security policy

## Supported code

Security fixes are made on the `main` branch. Historical artifacts under
`legacy/` are retained for auditability and are not supported runtime code.

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** form on the repository Security
tab. Do not open a public issue for an unpatched vulnerability or include live
credentials, private traffic captures, or identifying network data.

Include the affected commit, reproduction steps, impact, and any suggested
mitigation. Reports will be acknowledged as soon as practical; this is an
academic project without a guaranteed response SLA.

## Operational scope

This repository can capture traffic and alter firewall rules. Use it only on
systems and networks you own or are explicitly authorized to test. The default
dashboard is loopback-only, IPS enforcement is opt-in, and research-only models
are rejected by the live daemon. These controls reduce risk but do not make the
prototype suitable for unattended production security enforcement.

Never upload packet captures or logs before removing credentials, payloads,
addresses, hostnames, and other personal or confidential data.
