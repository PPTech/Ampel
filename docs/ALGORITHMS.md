# ALGORITHMS.md

Version: 0.6.0  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Core algorithms
1. `DB._migrate_db`
   - Detects missing schema fields and applies safe migration.
2. `DatasetBootstrapper.sync_catalog`
   - Upserts dataset legal/provenance metadata into SQLite.
3. `LaneAwareResolver.resolve`
   - Selects lane-matching traffic lamp by confidence; fallback to memory.
4. `AlertEngine.evaluate`
   - Applies deterministic safety rules.
5. `process_stream`
   - Full runtime pipeline from frame parsing to event output.
6. `run_ab_test`
   - Compares alert behavior under two threshold variants.
7. `security_check`
   - AST scan for unsafe execution primitives.
8. `scripts/dataset_manager.py`
   - Dataset sync, online checks, manual registration, and training-plan generation.
9. `scripts/sync_latest_to_main.sh`
   - Fetch/merge automation plus post-sync health checks.
