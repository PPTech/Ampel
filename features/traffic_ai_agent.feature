# Version: 0.6.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Traffic AI Assist generated BDD
  Scenario: Dataset schema migration
    Given existing sqlite database may miss usage_reason column
    When app initializes DB
    Then missing columns are migrated automatically

  Scenario: Demo mode validation
    Given free demo samples are available
    When demo mode executes
    Then app emits traffic-lamp and warning events

  Scenario: Dashboard access
    Given serve mode is active
    When user opens dashboard
    Then menu, map, camera block, and demo controls are visible
