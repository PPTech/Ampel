# Changelog

## [0.9.12] - 2026-02-16
### Added
- PR-6 monorepo predictive test selector `scripts/pts_selector.py` with shared/mobile/gadget target routing.
- PR-6 unified CI workflow `.github/workflows/main_pipeline.yml` with PTS, logic verification, Android, and gadget jobs.
- PR-6 failure diagnostics utility `scripts/diagnose_failure.py` generating redacted `failure_report.zip` artifacts.
- PR-6 one-command environment bootstrap `setup_dev_env.sh` for Python, Docker, Android SDK guidance, and pre-commit setup.
- BDD features `features/monorepo_pipeline.feature` and unit tests `tests/unit/test_pts_selector.py`.
- Upload-format regression tests `tests/unit/test_upload_image_formats.py`.

### Changed
- Improved `/ops/analyze-image` flow to accept and process broad photo formats (jpg/jpeg/png/gif/bmp/webp/tiff/heic/heif/avif) and return explicit format/error feedback.

## [0.9.11] - 2026-02-16
### Added
- PR-5 Android release hardening rules in `mobile/android/app/proguard-rules.pro` for obfuscation, CameraX/TFLite keeps, and stripping `Log.d`/`Log.v` calls.
- PR-5 runtime tamper checks in `mobile/android/app/src/main/java/com/pptech/ampel/security/IntegrityManager.kt` including signature SHA-256 verification and Knox/TIMA-capability probing.
- PR-5 GDPR encrypted preference layer in `mobile/android/app/src/main/java/com/pptech/ampel/data/SecureStorage.kt` using `EncryptedSharedPreferences` + `MasterKey`.
- PR-5 privacy kill-switch in `mobile/android/app/src/main/java/com/pptech/ampel/privacy/PrivacyManager.kt` for local data erase and edge-only network/domain controls.
- BDD feature `features/security_privacy_hardening.feature` for integrity, erase, and edge-only policy behavior.

### Changed
- Android module gradle config now includes `androidx.security:security-crypto` and BuildConfig signature placeholder for integrity validation.

## [0.9.10] - 2026-02-16
### Added
- PR-4 Android Gradle Kotlin DSL app module skeleton (`mobile/android/app/build.gradle.kts`) with CameraX, TFLite, Compose, and Accompanist dependencies.
- PR-4 Android manifest scaffold (`mobile/android/app/src/main/AndroidManifest.xml`) with camera/location/internet permissions.
- PR-4 CameraX pipeline manager (`mobile/android/app/src/main/java/com/pptech/ampel/camera/CameraManager.kt`) using background `ExecutorService` and VGA target resolution (640x480) for thermal safety.
- PR-4 TFLite wrapper (`mobile/android/app/src/main/java/com/pptech/ampel/ai/TrafficLightDetector.kt`) with NNAPI delegate preference and GPU fallback.
- PR-4 Compose UI (`mobile/android/app/src/main/java/com/pptech/ampel/ui/MainScreen.kt`) with `PreviewView`, detection overlay, and status banner.
- BDD feature for Android pipeline behavior (`features/android_mvp_camera_pipeline.feature`).

## [0.9.9] - 2026-02-16
### Added
- PR-3 TFLite-first detector adapter `src/core/vision/detector.py` implementing mobile-friendly inference plumbing with fallback mode.
- PR-3 anti-flicker temporal filter `src/core/vision/smoothing.py` with 5-frame sliding buffer and >3-frame persistence gate.
- PR-3 lane heuristic estimator `src/core/vision/lane_estimator.py` for LEFT/CENTER/RIGHT association using bbox center thresholds.
- PR-3 safety test `tests/unit/test_temporal_smoothing.py` and BDD feature `features/detection_smoothing.feature`.

### Changed
- `proto/python/traffic_ai_assist.py` HAL smoke path now uses `TFLiteTrafficDetector` + smoothing + lane heuristic prior to rules-engine evaluation.

## [0.9.8] - 2026-02-16
### Added
- PR-2 deterministic BDD feature file `features/traffic_rules.feature` for RED/GREEN/YELLOW rule outcomes.
- PR-2 deterministic logic engine `src/core/logic/rules_engine.py` including TTC calculation.
- PR-2 unit tests `tests/unit/test_rules_engine.py` and BDD bindings `tests/bdd/test_traffic_rules_bdd.py`.

### Changed
- `proto/python/traffic_ai_assist.py` now integrates `TrafficRulesEngine` in HAL smoke path to avoid hardcoded alert branching.

## [0.9.5] - 2026-02-15
### Added
- Added demo upload analysis endpoints (`/ops/analyze-image`, `/ops/analyze-video`) and dashboard upload controls for photo/video inspection.
- Added HAL scaffolding for dual deployment: Samsung CameraX/NPU delegate and Linux V4L2 provider plus CUDA runtime Dockerfile.
- Added security defense-in-depth scaffolding for tamper detection, Android Keystore/TEE key generation, certificate pinning, and edge-only upload firewall.
- Added contextual decision pseudo-code (`ContextAwareDecisionMaker`) and workflow `main.yml` for strict safety CI gates.

### Changed
- Dashboard map now prefers real GPS coordinates (browser geolocation or sample frame GPS) over route names.
- Developer/vision UX now states detection method explicitly (model + confidence + redundancy).

## [0.9.4] - 2026-02-15
### Added
- Added ISO-26262 style temporal consistency filter (`StateBuffer`) for traffic-light alert validation in `traffic_light_detector.py`.
- Added fail-safe "Check Traffic Light" warning when no light is detected for >5s at intersections.
- Added debug camera injection scaffolding (`CameraSourceInjector`) with looped video playback mode and GPX trace generator script.
- Added Android release security artifacts: `proguard-rules.pro`, `SecureMemoryCleaner.kt`, and Compose debug overlay.
- Added iOS release hardening and security scaffolding: `Release-Hardening.xcconfig`, `SecureMemoryCleaner.swift`, and SwiftUI debug overlay.
- Added BDD feature `features/temporal_consistency_filter.feature` and NGC model evaluation document.

### Changed
- Dashboard warning panel now uses state-aware colors (green/yellow/red) and event text color follows light state.
- Dataset statistics now calculate fallback sample counts from bundled demo samples, preventing `known samples: 0` when manifests have unknown counts.

## [0.9.2] - 2026-02-15
### Added
- Added MIT `LICENSE` and `ATTRIBUTION.md` for legal/compliance tracking.
- Added TensorFlow Lite detector wrapper `ai_engine/inference/traffic_light_detector.py` (permissive license path).
- Added `requirements.txt` and iOS `Podfile` with TensorFlowLiteSwift dependency.
- Added step-by-step dataset download guide file generation (`data/local_datasets/DOWNLOAD_STEPS.md`).

### Changed
- Developer Mode now shows styled detection cards with object color differentiation and bounding-box overlays instead of raw JSON-only output.
- Core model catalog now uses permissive options (COCO-SSD + EfficientDet-Lite0) and avoids AGPL core usage.
- Reorganized menu UX with settings entry and richer operations panel.

### Fixed
- Expanded health payload used by UI and monitoring to include runtime counters and model profile state.

## [0.9.0] - 2026-02-15
### Added
- Added free core AI model usage in Developer Mode: COCO-SSD bounding boxes + labels.
- Added `features/settings_menu.feature` for settings operations BDD coverage.
- Expanded health endpoint payload with uptime, dataset counters, model info, and trained threshold.
- Added visual Settings page with action buttons to run import/train/check operations and show results in UI.
- Updated demo random route handling to preserve sample route (berlin-city-center remains default in bundled samples).
- Improved Developer Mode to single camera preview and COCO-SSD object recognition with fallback heuristic.
- Added `scripts/import_all_datasets_local.sh` to materialize local metadata stubs for every listed external dataset source.
- Removed non-project optional installer command from README and replaced runbook with user-friendly command tables.
- Added `AdaptiveAgent` and `route_memory` schema with on-device-only federated learning logic (`ai_engine/inference/adaptive_agent.py`).
- Added setup automation `scripts/setup_project.sh` and QA runner `scripts/test_runner.py`.
- Added BDD feature files `features/federated_agent.feature` and `features/setup_test_suite.feature`.
- Replaced binary demo image with text-based preview in README to avoid PR binary-file limitations.
- Added developer mode manual camera Start/Stop controls and live object guess view.
- Added dataset selector support in random demo endpoint and dashboard controls.
- Added `--train-agent` learning command to persist learned threshold profile.
- Added additional feature files for BDD coverage (`developer_mode`, `dataset_pipeline`, `sync_workflow`).
- Added GitHub workflows `Python application` and `Python package` to address failing check names.
- Added Clean Architecture mobile scaffold plan in `docs/MOBILE_CLEAN_ARCHITECTURE.md` for Swift/Kotlin/Python boundaries and Mapbox navigation logic.
- Added module placeholders `ios/README.md`, `android/README.md`, and `ai_engine/README.md`.
- Added strict Cucumber file `features/traffic_light_core.feature` with requested core safety/privacy scenarios.
- Added `ai_engine/inference/edge_detection.py` (typed edge CV, LaneContextFilter, anonymizeFrame).
- Added CarPlay snippets `CarPlaySceneDelegate.swift` and `CarPlayAlertManager.swift` using CarPlay + MapboxNavigation integration strategy.
- Updated `/architecture` to HTML UX with current demo constraints and monitoring notes.

### Changed
- Improved dashboard visual UX and map/lamp overlay interaction.
- Expanded dataset manifest coverage and regenerated compiled manifest/training plan.
- Expanded documentation with install instructions and stronger GitHub-friendly structure.

### Security
- Kept AST security check and AB/security pre-release checks.

## [0.8.0] - 2026-02-15
- Added developer mode and dataset-selectable visual demo pipeline.
- Expanded dataset integration source list and visual pages.

## [0.7.0] - 2026-02-15
- Added visual menu/dashboard UX and random lamp simulation.
- Added dataset compile/dedup pipeline.

## [0.6.0] - 2026-02-15
- Added external dataset orchestration and main-sync automation scripts.

## [0.5.0] - 2026-02-15
- Added dashboard workflow hardening, governance docs, and release structure.
