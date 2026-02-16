# Version: 0.9.10
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Android MVP CameraX + TFLite pipeline
  As a Samsung-first mobile traffic assistant
  I want on-device camera analysis with low thermal load
  So that I get real-time light status safely

  Scenario: Camera analyzer runs at VGA on background executor
    Given CameraX uses an ImageAnalysis pipeline
    When CameraManager binds preview and analyzer
    Then analyzer uses a single-thread background executor
    And target resolution is 640x480 for thermal safety

  Scenario: Detector prefers NNAPI and falls back to GPU
    Given TrafficLightDetector loads a TFLite model
    When a delegate is selected
    Then NNAPI delegate is preferred
    And GPU delegate is used as fallback
