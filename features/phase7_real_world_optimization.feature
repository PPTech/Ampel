# Version: 0.9.14
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Real-world optimization and UX polish
  As a road user
  I want robust alerts and resilient UX in real conditions
  So that safety behavior remains stable in music playback, tunnels, and strong light changes

  Scenario: Audio ducking during alert playback
    Given media playback is active in another app
    When a critical traffic alert is played
    Then audio focus requests transient may-duck and releases after alert completion

  Scenario: Tunnel mode suppresses false stop-and-go
    Given GPS accuracy degrades above 50 meters for more than 2 seconds
    And accelerometer still detects movement
    When speed is requested for rules
    Then last known speed is maintained and stop-go alert is suppressed

  Scenario: YouTube link and media upload overlays
    Given user uploads media or provides a YouTube link
    When analyzer response is received
    Then dashboard overlays traffic sign/light boxes with reaction messages over media in real time
