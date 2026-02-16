# Version: 0.9.17
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Feature: Photo and video overlay detections in dashboard
  As a driver-assist user
  I want uploaded media to replace camera preview and show detection overlays
  So that traffic sign/light status is visible with reaction messages

  Scenario: Photo upload renders overlay boxes and color reaction
    Given a user uploads a photo in supported formats
    When the dashboard analyzes the image
    Then the photo is shown in the media stage
    And traffic sign and traffic light boxes are rendered
    And color reaction text is displayed

  Scenario: Video upload renders timeline overlays and messages
    Given a user uploads a video file
    When the dashboard analyzes the video
    Then the video is shown in the media stage
    And timeline detection boxes are drawn over playback
    And each frame message is displayed over video


  Scenario: User feedback improves future predictions
    Given a user reviewed a detection result
    When the user marks prediction as wrong and submits corrected state
    Then the app stores feedback bias for future image/video inference

  Scenario: YouTube analysis does not keep stale boxes from previous sources
    Given a user previously analyzed a local video with overlays
    When the user analyzes a YouTube link
    Then the dashboard clears stale local boxes
    And the message stream follows the YouTube timeline states
