# Changelog

## [0.4.0] - 2026-02-15
### Added
- Added free, license-safe synthetic sample dataset definitions and database seeding workflow.
- Added `demo_sample_frames` table and `--demo-mode` for instant end-to-end demonstration.
- Added `--export-demo-sample` command to create `data/demo_frames.jsonl`.
- Added dataset usage rationale (`usage_reason`) to external catalog schema.

### Changed
- Updated project metadata/documentation to include owner attribution, dataset license notes, and demo workflow.

### Security
- Kept parameterized SQL writes and strict privacy mode behavior for logs.
