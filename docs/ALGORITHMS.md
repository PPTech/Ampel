# ALGORITHMS.md

Version: 0.9.7  
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

10. `ai_engine/inference/edge_detection.py`
   - `LaneContextFilter`: use GPS heading + IMU-based lane estimate to keep only lane-relevant traffic lights.
   - `anonymizeFrame`: detect faces/plates and blur regions before inference/storage.
11. `CarPlayAlertManager` (Swift)
   - consumes alert states and applies high-priority audio + map banner update with audio-only fallback.

12. `AdaptiveAgent` (route_memory)
   - Predict green-light timing from `intersection_hash` + local time and update local weights from false-positive brake feedback.
13. `scripts/test_runner.py`
   - Runs BDD, benchmark gate (<50ms), and privacy audit scans.

14. `scripts/import_all_datasets_local.sh`
   - Creates local metadata stub folders for all manifest datasets with license/source context.
15. Developer mode COCO-SSD path
   - Browser loads COCO-SSD for multi-object recognition, falls back to color heuristic when unavailable.

16. Settings operations API
   - UI buttons trigger sync/import/train/ab/security endpoints and render JSON result panel.
17. Expanded health payload
   - Reports uptime, dataset/demo counts, and active model profile status for runtime observability.

18. `ai_engine/inference/traffic_light_detector.py`
   - TensorFlow Lite EfficientDet-Lite0 wrapper for permissive-license edge detection runtime.

19. Safe logger wrapper
   - Redacts GPS coordinates with regex and prevents UIImage/Bitmap/bytes serialization to persisted logs.

20. Clear data erasure flow
   - Settings action deletes local DB tables and local log artifacts for privacy rights handling.

21. Secrets bootstrap workflow
   - `scripts/setup_mobile_secrets.sh` creates git-ignored Android/iOS token files and enforces config injection.


## Temporal Consistency Filter (ISO 26262-oriented)

1. Collect frame-level detections (`red_light`, `green_light`) with confidence threshold > 0.85.
2. Push accepted states into a bounded `StateBuffer(size=3)`.
3. Emit `Valid Alert` only when the same state persists for 3 consecutive frames (~100ms at 30 FPS).
4. If no light is seen for > 5s while context indicates intersection, emit fail-safe visual warning: `Check Traffic Light`.
5. Otherwise remain in `Scanning...` state.


## PR-4 Android MVP Runtime Pipeline

1. Camera stream acquisition uses CameraX `Preview + ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST`.
2. Analyzer runs on a dedicated single-thread executor for deterministic latency and UI isolation.
3. Frames are resized and normalized for TensorFlow Lite model input.
4. Delegate strategy attempts NNAPI first (Samsung NPU path), then GPU fallback if NNAPI is unavailable.
5. Top detection class maps to app-level light state label and confidence.
6. Compose overlay renders inferred bounding boxes and a status banner for driver feedback.
