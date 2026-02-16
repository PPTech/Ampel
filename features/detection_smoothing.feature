# Version: 0.9.9
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Detection smoothing anti-flicker
  As a safety-critical traffic assistant
  I want single-frame glitches filtered out
  So that false alerts are reduced

  Scenario: Reject one-frame green glitch while red remains stable
    Given a 5-frame temporal smoothing buffer
    And current stable state is RED
    When one GREEN frame appears between RED frames
    Then output state remains RED
