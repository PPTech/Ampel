# Version: 0.9.19
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Professional dataset governance and color-safe alerting
  Scenario: Import only compatible datasets
    Given dataset metadata with license text
    When user imports a dataset from Dataset Manager
    Then only license-compatible datasets are accepted into active catalog

  Scenario: Delete dataset from active catalog
    Given a dataset exists in active catalog
    When user requests delete from Dataset Manager
    Then dataset metadata and related demo frames are removed

  Scenario: Color message follows detected lamp state
    Given media contains a traffic light region
    When dashboard estimates region color state
    Then alert message matches red/yellow/green state mapping
