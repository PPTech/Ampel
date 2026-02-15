# MEMORY.md

Version: 0.9.2  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Persistent project memory
- Product direction: visual-first traffic safety assistant.
- Core loop: dataset-driven scenario -> lane-aware inference -> alert action -> feedback/memory.
- Learning direction: maintain rule-based safety baseline, then train/refine detector thresholds and model profile.
- Data governance: store legal metadata for all external datasets before any download/train step.
- Release governance: every version must update README/CHANGELOG/MEMORY/BDD/features/workflows.

- Federated memory module: `AdaptiveAgent` predicts local time-to-green using intersection hash only and updates local weights from driver feedback.

- Local dataset metadata mirror script keeps all requested sources represented offline without redistributing restricted binaries.

- Settings page provides one-click UI actions for import/train/AB/security/health operations to reduce CLI friction.

- Permissive license model policy: core detector path uses Apache-2.0 compatible models (EfficientDet-Lite0 / COCO-SSD).

- Secrets policy: API tokens are injected from local config files (`android/local.properties`, `ios/Config.xcconfig`) and never hardcoded.

- Safe logging policy redacts geolocation coordinates and blocks image/bitmap payload serialization.

- Settings includes one-click Clear Data operation for GDPR Right to Erasure.
