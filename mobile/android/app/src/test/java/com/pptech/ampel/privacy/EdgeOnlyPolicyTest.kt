/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.privacy

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EdgeOnlyPolicyTest {
    @Test
    fun allowsOnlyAllowlistedHostWithoutRawPayload() {
        assertTrue(
            PrivacyManager.EdgeOnlyPolicy.isAllowed(
                host = "localhost",
                allowedDomains = setOf("localhost", "127.0.0.1"),
                isRawMediaPayload = false,
            ),
        )
        assertFalse(
            PrivacyManager.EdgeOnlyPolicy.isAllowed(
                host = "example.com",
                allowedDomains = setOf("localhost", "127.0.0.1"),
                isRawMediaPayload = false,
            ),
        )
    }

    @Test
    fun blocksRawMediaEvenForAllowlistedHost() {
        assertFalse(
            PrivacyManager.EdgeOnlyPolicy.isAllowed(
                host = "localhost",
                allowedDomains = setOf("localhost"),
                isRawMediaPayload = true,
            ),
        )
    }
}
