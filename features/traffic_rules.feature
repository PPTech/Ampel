# Version: 0.9.8
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Deterministic traffic rules engine (ISO 26262 oriented)
  Scenario: Red Light Violation Risk
    Given state is RED
    And speed is > 30 km/h
    And distance_to_stopline is < 20m
    Then output CRITICAL_ALERT

  Scenario: Green Light Idle (Stop & Go)
    Given state is GREEN
    And speed is 0 km/h for > 3 seconds
    Then output INFO_ALERT ("Please go")

  Scenario: Yellow Light Caution
    Given state is YELLOW
    Then output WARN_ALERT
