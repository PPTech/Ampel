# Version: 0.9.2
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Secrets and privacy guardrails
  Scenario: Mobile secrets are not hardcoded
    Given Android and iOS apps require MAPBOX access token
    When project is configured for local development
    Then tokens are read from local git-ignored config files

  Scenario: Safe logger redacts sensitive fields
    Given a log payload contains GPS-like coordinates
    When Safe logger serializes the payload
    Then coordinates are replaced with [GEO-REDACTED]
    And bitmap or image payloads are redacted

  Scenario: Right to erasure action
    Given user opens settings
    When user clicks Clear All Local Data
    Then local databases and logs are wiped immediately
