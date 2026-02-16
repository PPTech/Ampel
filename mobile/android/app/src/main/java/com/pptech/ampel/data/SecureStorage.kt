/*
Version: 0.9.11
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * GDPR-safe encrypted preference store.
 * Sensitive values must never be persisted in plaintext SharedPreferences.
 */
class SecureStorage(context: Context) {
    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val prefs: SharedPreferences = EncryptedSharedPreferences.create(
        context,
        FILE_NAME,
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    fun setAlertVolume(value: Int) {
        prefs.edit().putInt(KEY_ALERT_VOLUME, value.coerceIn(0, 100)).apply()
    }

    fun getAlertVolume(defaultValue: Int = 70): Int {
        return prefs.getInt(KEY_ALERT_VOLUME, defaultValue.coerceIn(0, 100))
    }

    fun setHomeLocationHash(hash: String) {
        prefs.edit().putString(KEY_HOME_LOCATION_HASH, hash).apply()
    }

    fun getHomeLocationHash(): String? = prefs.getString(KEY_HOME_LOCATION_HASH, null)

    fun clearAll() {
        prefs.edit().clear().apply()
    }

    companion object {
        private const val FILE_NAME = "ampel_secure_prefs"
        private const val KEY_ALERT_VOLUME = "alert_volume"
        private const val KEY_HOME_LOCATION_HASH = "home_location_hash"
    }
}
