// Version: 0.9.0
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

import AVFoundation
import CarPlay
import Foundation
import MapKit

final class CarPlayAlertManager {
    enum AlertState: String {
        case redLightSpeeding = "RED_LIGHT_SPEEDING"
        case greenWait = "GREEN_WAIT"
        case none = "NONE"
    }

    private let speech = AVSpeechSynthesizer()

    func handle(state: AlertState, mapTemplate: CPMapTemplate?) {
        switch state {
        case .redLightSpeeding:
            configureHighPriorityAudioSession()
            speak("Warning. Red light ahead. Reduce speed immediately.")
            updateBanner(on: mapTemplate, message: "RED LIGHT + SPEEDING")
        case .greenWait:
            configureHighPriorityAudioSession()
            speak("Traffic light is green. Move if safe.")
            updateBanner(on: mapTemplate, message: "GREEN LIGHT: GO IF SAFE")
        case .none:
            break
        }
    }

    func configureHighPriorityAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
            try session.setActive(true, options: [.notifyOthersOnDeactivation])
        } catch {
            // In production, log to telemetry with privacy-safe policy.
            print("Audio session setup failed: \(error)")
        }
    }

    func updateBanner(on mapTemplate: CPMapTemplate?, message: String) {
        guard let mapTemplate else { return }
        let trip = CPTrip(origin: MKMapItem.forCurrentLocation(), destination: MKMapItem.forCurrentLocation(), routeChoices: [])
        let maneuver = CPManeuver()
        maneuver.instructionVariants = [message]
        let travel = CPTravelEstimates(distanceRemaining: 0, timeRemaining: 0)
        mapTemplate.updateEstimates(travel, for: trip)
        mapTemplate.showTripPreviews([trip], textConfiguration: CPTripPreviewTextConfiguration(startButtonTitle: "OK", additionalRoutesButtonTitle: nil, overviewButtonTitle: nil))
        _ = maneuver
    }

    func triggerAudioOnlyFallback(for state: String) {
        configureHighPriorityAudioSession()
        speak("Traffic signal alert. \(state).")
    }

    private func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.48
        utterance.volume = 1.0
        speech.speak(utterance)
    }
}
