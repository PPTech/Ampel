# Changelog

## [0.9.0] - 2026-02-15
### Added
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
- Updated `/architecture` to HTML UX and locked demo route to `berlin-city-center` during dataset switching.

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
