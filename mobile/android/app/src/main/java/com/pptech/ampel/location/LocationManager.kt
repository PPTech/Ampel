/*
Version: 0.9.14
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.location

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import java.util.concurrent.TimeUnit

class LocationManager(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private var lastLocation: Location? = null
    private var lastLocationTsMs: Long = 0L
    private var lastKnownSpeedMps: Float = 0f
    private var motionMagnitude: Float = 0f

    init {
        accelerometer?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    fun updateLocation(location: Location) {
        lastLocation = location
        lastLocationTsMs = System.currentTimeMillis()
        if (location.hasSpeed()) {
            lastKnownSpeedMps = location.speed
        }
    }

    fun tunnelModeActive(nowMs: Long = System.currentTimeMillis()): Boolean {
        val stale = nowMs - lastLocationTsMs > TimeUnit.SECONDS.toMillis(2)
        val poorAccuracy = (lastLocation?.accuracy ?: Float.MAX_VALUE) > 50f
        val moving = motionMagnitude > 0.8f || lastKnownSpeedMps > 1.5f
        return (stale || poorAccuracy) && moving
    }

    fun speedForRules(nowMs: Long = System.currentTimeMillis()): Float {
        return if (tunnelModeActive(nowMs)) lastKnownSpeedMps else (lastLocation?.speed ?: 0f)
    }

    fun shouldSuppressStopAndGoAlert(nowMs: Long = System.currentTimeMillis()): Boolean {
        return tunnelModeActive(nowMs)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type != Sensor.TYPE_ACCELEROMETER) return
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]
        motionMagnitude = kotlin.math.sqrt((x * x + y * y + z * z).toDouble()).toFloat() / SensorManager.GRAVITY_EARTH
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    fun release() {
        sensorManager.unregisterListener(this)
    }
}
