# ATTRIBUTION

Version: 0.9.4  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Open-source libraries used

| Library | License | Usage |
|---|---|---|
| TensorFlow.js COCO-SSD | Apache-2.0 | Browser-side object detection in Developer Mode |
| TensorFlow Lite (EfficientDet-Lite0 target) | Apache-2.0 | Core edge detector runtime target |
| OpenCV | Apache-2.0 | Privacy preprocessing (face/plate blur) |
| Python stdlib (`sqlite3`, `http.server`, etc.) | PSF | Local APIs and runtime orchestration |
| Mapbox Navigation SDK (planned mobile runtime) | Mapbox terms | CarPlay/AAOS navigation integration path |

## Removed / avoided due license policy

- `ultralytics` / YOLOv8 runtime dependency is not used as the core app engine in this branch to avoid AGPL-3.0 licensing risk for proprietary distribution.
