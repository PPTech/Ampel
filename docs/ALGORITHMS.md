# ALGORITHMS.md

Version: 0.9.21  
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


## PR-5 Security/Privacy Runtime Flow

1. On app start, integrity manager computes signing certificate SHA-256 and compares with approved release hash.
2. If mismatch is detected, runtime raises `SecurityException` and safety-critical AI path is disabled.
3. SecureStorage persists sensitive settings using Android Keystore-backed `MasterKey` and encrypted shared preferences.
4. Privacy manager enforces edge-only outbound policy: non-whitelisted domains or raw media payload attempts are blocked.
5. GDPR kill-switch erases secure prefs, shared prefs, databases, cache, and log artifacts on demand.


## PR-6 Monorepo CI Orchestration Flow

1. `pts_selector.py` evaluates changed paths and emits platform-job flags.
2. Workflow executes logic verification always for deterministic rule consistency.
3. Android build/tests run only when `mobile/**` or shared contracts change.
4. Gadget docker/integration tests run only when `gadget/**` or shared contracts change.
5. On failures, `diagnose_failure.py` packages failed scenario hints, redacted logs, and git diff into `failure_report.zip`.
6. `/ops/analyze-image` decodes base64 payloads, detects file signatures (jpeg/png/gif/bmp/webp/tiff/heic/heif/avif), and returns structured detection response/error.


## PR-6.1 Media Overlay Interaction Flow

1. On photo upload, dashboard stops camera stream and switches media stage to image mode.
2. Image analyzer returns sign/light detections with bbox metadata.
3. Frontend overlay canvas projects normalized boxes and color-coded labels over image.
4. On video upload, dashboard switches to video mode and replays timeline detections.
5. Per-frame message (STOP/GO/CAUTION) is rendered directly over video surface for immediate feedback.


## Phase 7 Runtime Safety Optimization Flow

1. AlertManager requests `AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK` before siren/TTS and abandons focus on completion.
2. LocationManager marks tunnel mode when GPS is stale (>2s) or inaccurate (>50m) while accelerometer indicates movement.
3. Tunnel mode holds last known speed and suppresses false stop-go prompts.
4. TFLiteDelegateFactory chooses delegate by Samsung chipset (Exynos->GPU, Snapdragon->NNAPI) and falls back to CPU on initialization exceptions.
5. MainScreen adapts rendering by ambient lux for true-black night mode and high-contrast bright-sun mode.
6. Dashboard media mode pipeline now supports photo/video/YouTube with frame-message overlays and traffic sign/light boxes.


## Phase 7.1 CI and Hygiene Control Flow

1. PTS maps changed files to platform test scopes.
2. Hygiene-fast stage always runs lint, formatting, and py_compile.
3. Runtime artifact scanner blocks checked-in logs/db/zip artifacts.
4. Brain tests run unit + BDD + self/security checks.
5. Android and gadget jobs run only when selected by PTS.


## Phase 7.2 Dashboard and Developer Detection Reliability

1. `dashboard_html` media state machine
   - Switches a single visual stage between camera/photo/video modes and keeps overlays synced to active source.
2. Camera permission fallback loop
   - Detects permission/runtime errors and surfaces deterministic retry guidance instead of silent blank preview.
3. YouTube timeline reaction bridge
   - Converts URL-derived analysis timeline into per-frame reaction text + object box overlays.
4. Developer mode temporal smoothing
   - Aggregates recent labels and suppresses one-frame spikes for steadier object cards and overlays.


## Phase 7.3 Overlay Accuracy and Feedback Learning

1. Contain-fit bbox projection
   - Computes rendered media rectangle inside canvas (`object-fit: contain`) and projects normalized model boxes into that rectangle.
2. Overlay lifecycle reset
   - Clears canvas and stops timeline loop on mode transitions to prevent stale detections from previous media sources.
3. Stable timeline state model
   - Derives a dominant traffic-light state per video source and keeps timeline frames consistent to avoid random color flips.
4. Human feedback memory
   - Stores local correctness feedback in `model_profile` and applies bias to subsequent image/video state defaults.


## Phase 7.4 Browser-Assisted Media Detection and CI Compatibility

1. Browser COCO-SSD assist
   - Loads on dashboard and runs lightweight client inference for photo/video/youtube media elements.
2. Temporal state smoothing (client)
   - Aggregates last N states and chooses the dominant state for user-visible stability.
3. Dynamic confidence variance
   - Uses payload-derived digest variance so fallback confidences are not hard-coded constants.
4. Compatibility CI gates
   - Provides explicit migration workflows (`main_pipeline`, `main/safety-ci`) to satisfy protected-branch required checks.


## Phase 7.5 Professional Dataset Governance and Color-State Inference

1. License compatibility gate
   - Catalog sync/import validates license text against allowed compatibility keywords.
2. Dataset lifecycle API
   - Supports list/import/delete for active datasets used by runtime and UI.
3. Browser ROI color-state estimator
   - Reads detected traffic-light region pixels and derives red/yellow/green state from channel dominance.
4. Professional operations UX
   - Settings and dataset views grouped by workflow (diagnostics, AI ops, data governance, privacy).


## Phase 8 Android Frame Preprocessing and Release Hardening

1. `YUV_420_888 -> RGB` conversion
   - Converts camera planes to NV21 and decodes to RGB bitmap for analyzer input.
2. Structured classifier output contract
   - For classifier-only TFLite outputs, emit `bbox=null`, include `timestamp`, and smooth state transitions via temporal buffer.
3. Release signature gate
   - Gradle release build fails if `OFFICIAL_SIG_SHA256` is not injected from CI/property.
4. Centralized edge-only network policy
   - All URL checks pass through `PrivacyManager.EdgeOnlyPolicy` (allowlist + raw media block).

14. `dataset_name_matches` + `/demo/random`
   - Supports dataset alias matching (e.g., "LISA traffic-light reference" -> LISA dataset family) to ensure requested dataset demos are actually loaded.
15. `_video_timeline` + `analyze_youtube_link`
   - Builds state progression timelines with color-aware messages and optional bbox suppression for metadata-only sources (YouTube), preventing stale overlays.
