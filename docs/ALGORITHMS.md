# ALGORITHMS.md

Version: 0.7.0  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Key algorithms
1. `DatasetRegistry.datasets`: load external manifest and normalize rows.
2. `DatasetBootstrapper.sync_catalog`: upsert legal/provenance fields to DB.
3. `LaneAwareResolver.resolve`: lane-specific light selection with memory fallback.
4. `AlertEngine.evaluate`: deterministic safety rules.
5. `process_stream`: frame -> inference(optional) -> resolver -> alert -> event.
6. `dashboard_html` + `/demo/random`: random dataset sample visualized on map canvas, then reacted by agent.
7. `scripts/dataset_manager.py::compile_manifest`: merge + dedup + optimize dataset list for project usage.
8. `scripts/sync_latest_to_main.sh`: merge/sync flow with post-sync A/B + security checks.
