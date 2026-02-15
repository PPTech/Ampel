// Version: 0.9.0
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
// Author: Dr. Babak Sorkhpour with support from ChatGPT

import CarPlay
import Foundation
import UIKit
import MapboxNavigation

final class CarPlaySceneDelegate: UIResponder, CPTemplateApplicationSceneDelegate {
    private let allowHeadUnitVideo = false // policy guard: no live camera feed on CarPlay
    private var interfaceController: CPInterfaceController?
    private var mapTemplate: CPMapTemplate?
    private let alertManager = CarPlayAlertManager()

    func templateApplicationScene(
        _ templateApplicationScene: CPTemplateApplicationScene,
        didConnect interfaceController: CPInterfaceController,
        to window: CPWindow
    ) {
        self.interfaceController = interfaceController

        let mapTemplate = CPMapTemplate()
        mapTemplate.mapDelegate = self
        self.mapTemplate = mapTemplate

        // Primary: map-centric navigation template.
        // Overlay restrictions: CarPlay may block arbitrary video layers.
        // We therefore keep map + navigation + banners and fallback to audio alerts.
        interfaceController.setRootTemplate(mapTemplate, animated: true)

        configureMapboxNavigationIfAvailable()
    }

    func templateApplicationScene(
        _ templateApplicationScene: CPTemplateApplicationScene,
        didDisconnect interfaceController: CPInterfaceController,
        from window: CPWindow
    ) {
        self.interfaceController = nil
        self.mapTemplate = nil
    }

    private func configureMapboxNavigationIfAvailable() {
        guard allowHeadUnitVideo == false else { return }
        // Example hook for Mapbox integration in production app:
        let token = MapboxConfig.accessToken
        _ = token
        // let core = MapboxNavigationProvider(coreConfig: .init(accessToken: token))
        // let nav = core.mapboxNavigation
        // mapTemplate?.startNavigationSession(for: mapTemplate!.trip)
    }

    func showTrafficAlertPOI(title: String, subtitle: String) {
        guard let interfaceController else { return }

        let poi = CPPointOfInterest(
            location: .init(latitude: 52.5200, longitude: 13.4050),
            title: title,
            subtitle: subtitle,
            summary: "Traffic signal warning",
            detailTitle: title,
            detailSubtitle: subtitle,
            detailSummary: "Audio guidance active",
            pinImage: nil
        )
        let poiTemplate = CPPointOfInterestTemplate(title: "AmpelAI Alerts", pointsOfInterest: [poi], selectedIndex: NSNotFound)
        interfaceController.pushTemplate(poiTemplate, animated: true)
    }

    /// Attempt to update map overlays if platform policies allow the template integration.
    /// If not allowed, keep banner + audio-only mode.
    func updateSignalOverlayOrFallback(state: String) {
        guard let mapTemplate else {
            alertManager.triggerAudioOnlyFallback(for: state)
            return
        }

        let item = CPMapTemplate.PanButton { [weak self] _ in
            self?.alertManager.triggerAudioOnlyFallback(for: state)
        }
        item.image = UIImage(systemName: "exclamationmark.triangle")
        mapTemplate.leadingNavigationBarButtons = [item]
        alertManager.updateBanner(on: mapTemplate, message: state)
    }
}

extension CarPlaySceneDelegate: CPMapTemplateDelegate {}
