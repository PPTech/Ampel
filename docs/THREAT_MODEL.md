# THREAT_MODEL.md

Version: 0.9.15  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Assets
- On-device camera frames (ephemeral by default)
- Alert decision outputs (safety-critical)
- Local settings and anonymized telemetry
- Model binaries and runtime integrity state

## Attackers
- Local attacker with filesystem access
- Network attacker attempting MITM on optional outbound metadata traffic
- Reverse engineer attempting tamper/re-signing

## Trust boundaries
- Device hardware/OS keystore boundary
- App process boundary
- Optional outbound API boundary (metadata-only)

## Mitigations (Defense in depth)
- Edge-only processing default; no raw frame upload paths.
- Encrypted local settings storage (Android Keystore / encrypted prefs).
- Signature/integrity verification at startup.
- Privacy erase + retention cleanup controls.
- CI hygiene + secrets checks + strict tests on core logic.

## Residual risk
- Physical compromise of rooted device can still extract runtime memory.
- Sensor spoofing can degrade perception quality.
