# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Dataset compile and sync pipeline
  Scenario: Compile and deduplicate manifest
    Given external and manual manifests exist
    When compile-manifest command runs
    Then compiled manifest is generated with totals

  Scenario: Sync legal metadata to DB
    Given compiled manifest is available
    When dataset sync command runs
    Then source/license/usage/status fields are stored
