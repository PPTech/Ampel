# Version: 0.9.4
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT

Feature: Safety critical traffic logic
  Scenario: Overspeed on red triggers audio visual alert
    Given a red light and vehicle speed 55
    When the rules engine evaluates the frame
    Then alert key is "overspeed_red"

  Scenario: Red crossing triggers siren
    Given a red light and crossed stop line
    When the rules engine evaluates the frame
    Then alert key is "red_crossed"

  Scenario: Green and no movement triggers go alert
    Given a green light and stationary seconds 3
    When the rules engine evaluates the frame
    Then alert key is "green_wait"

  Scenario: Pedestrian fallback warning
    Given no critical light event and pedestrian detected
    When the rules engine evaluates the frame
    Then alert key is "pedestrian"

  Scenario: Ambiguous lane asks user selection
    Given lane context is ambiguous
    When lane filter cannot assign exactly one light
    Then user selection is required

  Scenario: Temporal buffer one frame remains scanning
    Given temporal detector has 1 consecutive red frame
    When temporal consistency is checked
    Then detector status is "Scanning..."

  Scenario: Temporal buffer two frames remains scanning
    Given temporal detector has 2 consecutive red frames
    When temporal consistency is checked
    Then detector status is "Scanning..."

  Scenario: Temporal buffer three frames validates alert
    Given temporal detector has 3 consecutive red frames
    When temporal consistency is checked
    Then detector status is "Valid Alert: red_light"

  Scenario: Fail-safe warning after detection timeout at intersection
    Given temporal detector has no signal for 6 seconds at intersection
    When temporal consistency is checked
    Then detector status is "Check Traffic Light"

  Scenario: Latency budget under 50ms
    Given synthetic detector input
    When inference latency is measured
    Then latency is below 50 milliseconds
