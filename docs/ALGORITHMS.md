# ALGORITHMS.md

Version: 0.9.0  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Algorithm map
1. `DatasetRegistry.datasets`
   - Load and normalize external dataset manifest rows.
2. `DatasetBootstrapper.sync_catalog`
   - Upsert legal/source/usage metadata in SQLite.
3. `LaneAwareResolver.resolve`
   - Resolve lane-specific light by confidence then memory fallback.
4. `AlertEngine.evaluate`
   - Apply safety rules in deterministic priority.
5. `train_agent_model`
   - Learn/update `red_speed_threshold_kph` from seeded telemetry and persist in `model_profile`.
6. `process_stream`
   - Parse frame -> optional model infer -> resolve -> evaluate -> event output.
7. Visual dashboard demo
   - Random sample fetch -> map/lamp visualization -> browser visual recognition -> event reaction.
8. Developer mode detector
   - Live camera + pixel statistics -> object guess labels.
9. `scripts/dataset_manager.py`
   - compile/dedup/sync/health/manual registration/training-plan generation.
10. `scripts/sync_latest_to_main.sh`
   - merge sync workflow + post-merge compile/safety checks.
