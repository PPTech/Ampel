# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Edge CV lane filtering and CarPlay alert projection
  Scenario: Lane-aware filtering keeps correct traffic lights
    Given detected lights include multiple lane candidates
    And gps heading and IMU telemetry are available
    When LaneContextFilter runs
    Then only lane-relevant traffic lights are kept

  Scenario: Privacy preprocessing anonymizes sensitive regions
    Given camera frame contains faces and license plates
    When anonymizeFrame is executed before inference
    Then face and plate regions are Gaussian blurred
    And raw non-anonymized frame is not stored remotely

  Scenario: CarPlay map overlay is restricted by OS policy
    Given CarPlay template limitations block direct video overlay
    When RED_LIGHT_SPEEDING is received
    Then CarPlayAlertManager triggers high-priority ducked audio
    And the map template is updated with a visual warning banner
    And the system falls back to audio-only if map overlay is blocked
