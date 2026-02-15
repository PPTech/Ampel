# Version: 0.4.1
# Code generated with support from CODEX and CODEX CLI
# Owner / Idea / Management: Dr. Babak Sorkhpour - https://x.com/Drbabakskr

Feature: Smart lane-aware traffic AI agent with dashboard and demo mode
  Scenario: Existing DB migration
    Given an old sqlite db without usage_reason column
    When app starts
    Then schema is migrated automatically without crash

  Scenario: Demo mode
    Given free synthetic sample frames are available
    When demo mode runs
    Then alert events are emitted and saved

  Scenario: Dashboard menus
    Given server is running
    When user opens /menu
    Then menu items are returned
