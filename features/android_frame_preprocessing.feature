# Version: 0.9.20
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Android production MVP preprocessing and hardening
  Scenario: Camera frame conversion produces non-black RGB
    Given a YUV_420_888 frame from CameraX analyzer
    When TrafficLightDetector converts it to RGB bitmap
    Then bitmap pixel variance is non-zero

  Scenario: Classifier-only detection emits structured payload
    Given TFLite classifier outputs only class scores
    When detector creates Detection model
    Then bbox is null and timestamp is present

  Scenario: Release build signature gate
    Given a release build is triggered
    When OFFICIAL_SIG_SHA256 is missing
    Then build fails before producing release artifact
