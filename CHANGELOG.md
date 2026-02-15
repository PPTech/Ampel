# Changelog

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
