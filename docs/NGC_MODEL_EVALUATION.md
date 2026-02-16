# Version: 0.9.3
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

# NVIDIA NGC Model Evaluation (Traffic/Lane/Vehicle)

## Scope
This document evaluates NVIDIA NGC catalog options requested for AmpelAI and defines a license-safe integration path.

## Findings
1. **NVIDIA Image Segmentation collection** (NGC) includes high-quality segmentation models, but artifacts are generally optimized for TensorRT/TAO pipelines and may require NGC account licensing acceptance.
2. **TAO VehicleTypeNet (`pruned_onnx_v1.1.0`)** is suitable for vehicle-type context enrichment, not primary traffic-light state classification.
3. For **mobile-first and permissive licensing baseline**, the current core remains:
   - TensorFlow Lite EfficientDet-Lite0 (Apache-2.0) for edge object detection.
   - Browser COCO-SSD (Apache-2.0) for developer mode preview.

## Decision
- Keep permissive primary detector path in app runtime.
- Add **optional NGC adapter pathway** only when model artifact and accepted license are provided locally by the operator.
- Do not redistribute NGC weights in this repository.

## Integration Checklist
- Download NGC artifact locally with account-bound token.
- Store model under local non-versioned path (gitignored).
- Convert/optimize for platform runtime (TensorRT for backend or ONNX Runtime for validation).
- Validate mAP/FPS vs EfficientDet-Lite0 baseline before enabling.

## Compliance Notes
- Respect NGC model-specific terms before any commercial deployment.
- Keep no raw camera upload policy unchanged.
- Continue GDPR/CCPA controls: redacted logs and right-to-erasure.
