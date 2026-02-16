// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

import Foundation

enum SecureMemoryCleaner {
    static func wipe(buffer: inout [UInt8]) {
        for idx in buffer.indices {
            buffer[idx] = 0
        }
    }
}
