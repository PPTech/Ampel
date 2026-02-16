# MEMORY.md

Version: 0.9.17  
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

- ISO 26262 temporal consistency filter now requires 3-frame persistence for red/green signal alerts and fail-safe warning after 5s without light near intersections.
- Debug stack now supports camera source injection (`MOCK_MODE`) with looped `test_drive_red_light.mp4` and synchronized GPX playback for non-driving QA.
- Mobile hardening assets added: ProGuard/R8 rules, release stripping config, and secure frame-buffer zeroization utilities.

- Dual deployment HAL scaffold added for Samsung NPU and Linux V4L2 runtime convergence.
- Dashboard now uses real GPS location when available for Google Maps embedding.

- PR-1 defines shared traffic_event schema + HAL interfaces so Android and Linux targets can reuse core decision flow without logic forks.

- PR-2 introduces deterministic TTC-based rule engine for safety-critical non-AI decisions.

- PR-3 introduces TFLite detector adapter, anti-flicker temporal buffer, and lane-estimation heuristics for mobile-safe vision pipeline.

- PR-4 Android MVP introduces CameraX + TFLite + Compose overlay stack under `mobile/android`.

- PR-5 adds Android integrity checks, encrypted preferences, edge-only enforcement, and local data nuke controls.

- PR-6 adds unified monorepo CI/CD with predictive test selection and self-healing diagnostics artifacts.
- Image upload analyzer now accepts broad image formats and reports explicit format-detection errors.

- Dashboard now prioritizes uploaded photo/video rendering with detection boxes and over-media reaction messages.

- Phase 7 adds audio ducking, tunnel-mode dead reckoning, Samsung delegate strategy, and lux-based UI adaptation for real-world stability.

- Dashboard now supports YouTube-link analysis timeline with over-media detection boxes and action text.

- Phase 7.1 consolidates CI into a single master-branch workflow with PTS and artifact hygiene enforcement.

- Threat model documentation added to formalize trust boundaries and defense-in-depth assumptions.

- Phase 7.2 dashboard reliability pass improved camera-permission fallback UX, media overlay resilience, and developer-mode detection stability.

- Phase 7.3 fixed contain-fit bbox alignment, stale overlay reset between media sources, and added on-device user-feedback learning memory for prediction bias tuning.
