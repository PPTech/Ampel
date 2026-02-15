Feature: Smart lane-aware traffic AI agent
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

  Scenario: Lane mismatch
    Given no lane-specific light exists
    When event is evaluated
    Then user selection is required
