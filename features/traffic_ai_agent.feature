# Version: 0.4.0
# Code generated with support from CODEX and CODEX CLI
# Owner / Idea / Management: Dr. Babak Sorkhpour - https://x.com/Drbabakskr

Feature: Smart lane-aware traffic AI agent with free demo dataset
  Scenario: Seed free sample data into database
    Given dataset catalog and free sample scenarios are available
    When demo mode is enabled
    Then demo frames are inserted into demo_sample_frames table

  Scenario: Overspeed under red light
    Given speed is above red threshold
    And lane matched light is red
    When event is evaluated
    Then audio warning is emitted

  Scenario: Crossing stop-line under red
    Given vehicle crossed stop line
    And lane matched light is red
    When event is evaluated
    Then siren warning is emitted

  Scenario: Pedestrian warning
    Given pedestrian is detected in current frame
    When event is evaluated
    Then visual caution warning is emitted
