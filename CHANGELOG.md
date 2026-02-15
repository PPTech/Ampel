# Changelog

## [0.6.0] - 2026-02-15
### Added
- Added `scripts/sync_latest_to_main.sh` to fetch/merge latest into main and run sync health checks.
- Added `scripts/dataset_manager.py` for dataset sync, online health checks, manual dataset registration, and training-plan generation.
- Added `data/external_datasets_manifest.json` with legal/usage metadata for requested external datasets.

### Changed
- Updated app dataset registry to load external dataset manifest dynamically.
- Updated project documentation and standards artifacts for release governance.

### Fixed
- Improved operational workflow to reduce serve-port and sync friction in daily use.
