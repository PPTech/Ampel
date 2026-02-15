# Version: 0.9.3
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Temporal consistency filter for traffic lights
  Scenario: Red alert requires 3 consecutive frames
    Given red light detections with confidence above 0.85
    When only 1 or 2 consecutive frames are seen
    Then detector returns "Scanning..."
    And no critical alert is emitted

  Scenario: Valid red alert on frame stability
    Given red light detections with confidence above 0.85
    When 3 consecutive frames are seen
    Then detector returns "Valid Alert: red_light"

  Scenario: Fail-safe warning near intersection
    Given no light is detected for more than 5 seconds
    And vehicle context indicates an intersection
    When temporal filter is evaluated
    Then detector returns "Check Traffic Light"
