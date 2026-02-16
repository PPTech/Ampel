# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Traffic AI Assist generated BDD
  Scenario: Agent training refinement
    Given seeded demo samples and stored telemetry
    When train-agent command runs
    Then learned model profile is updated for alert thresholds

  Scenario: Dataset schema migration
    Given existing sqlite database may miss usage_reason column
    When app initializes DB
    Then missing columns are migrated automatically

  Scenario: Random dataset demo reaction
    Given free demo samples are available
    When user requests random demo with optional dataset filter
    Then app visualizes lamp and emits warning event based on rules

  Scenario: Visual dashboard and menus
    Given serve mode is active
    When user opens menu and dashboard
    Then visual cards, map, camera block, and demo controls are visible

  Scenario: Developer mode live detection
    Given developer mode is opened
    When browser camera stream is active
    Then client-side object guesses are shown continuously
