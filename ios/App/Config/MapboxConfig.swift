// Version: 0.9.2
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

import Foundation

enum MapboxConfig {
    static var accessToken: String {
        guard let token = Bundle.main.object(forInfoDictionaryKey: "MAPBOX_ACCESS_TOKEN") as? String,
              !token.isEmpty else {
            fatalError("MAPBOX_ACCESS_TOKEN missing. Add it via Config.xcconfig and Info.plist mapping.")
        }
        return token
    }
}
