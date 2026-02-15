# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Traffic light core safety behavior
  In order to keep drivers safe at intersections
  As the AmpelAI assistant
  I want to detect signal state, lane relevance, and privacy requirements before issuing alerts

  Scenario: Red light detected + Speeding -> Audio/Visual Alert
    Given the active lane has a detected traffic light with state "red"
    And vehicle speed is above the configured red-light threshold
    When the traffic-light safety policy is evaluated
    Then the app must trigger an alert key "overspeed_red"
    And the app must use alert channels "audio" and "visual"

  Scenario: Green light + No movement -> Go Alert
    Given the active lane has a detected traffic light with state "green"
    And the vehicle has remained stationary for at least 5 seconds
    When the traffic-light safety policy is evaluated
    Then the app must trigger an alert key "green_wait"
    And the app must show message "Go if safe"

  Scenario: Ambiguous Lane -> User Input Prompt
    Given multiple traffic lights are detected and lane mapping confidence is low
    When the lane-to-signal resolver runs
    Then the app must request manual user signal selection
    And the unresolved event must be stored for learning feedback

  Scenario: Privacy Mode -> Blur Faces/Plates before processing
    Given privacy mode is enabled
    And camera frames include human faces or license plates
    When preprocessing runs before model inference
    Then face regions must be blurred
    And license-plate regions must be blurred
    And no raw frame may be uploaded to remote services
