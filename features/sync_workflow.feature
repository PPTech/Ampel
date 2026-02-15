# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Sync latest to main workflow
  Scenario: Merge and health checks
    Given remote and target branch are configured
    When sync_latest_to_main.sh runs
    Then merge occurs and AB/security/dataset checks execute
