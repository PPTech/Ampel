# Version: 0.9.1
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Settings menu operations UI
  Scenario: User runs dataset import from settings
    Given settings page is open
    When user clicks Import Datasets button
    Then local dataset metadata stubs are generated
    And operation output is shown in the UI panel

  Scenario: User runs train and checks from settings
    Given settings page is open
    When user clicks Train AI Model and Run A/B Test
    Then model training output is shown
    And A/B test report is shown in the UI panel

  Scenario: Health details are expanded
    Given service is running
    When user opens /health
    Then response includes uptime and dataset/model counters
