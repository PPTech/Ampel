# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Developer mode visual detection
  Scenario: Start and stop camera manually
    Given developer page is open
    When user clicks start camera button
    Then live stream starts
    When user clicks stop camera button
    Then live stream stops

  Scenario: Object guess detection
    Given camera stream is active
    When periodic frame analysis runs
    Then object guess labels are shown
