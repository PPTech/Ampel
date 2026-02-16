/*
Version: 0.9.14
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ai

import android.os.Build
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

object TFLiteDelegateFactory {
    data class DelegateBundle(
        val options: Interpreter.Options,
        val nnApiDelegate: NnApiDelegate? = null,
        val gpuDelegate: GpuDelegate? = null,
    )

    fun createOptions(): DelegateBundle {
        val manufacturer = Build.MANUFACTURER.lowercase()
        val hardware = Build.HARDWARE.lowercase()
        val isSamsung = manufacturer.contains("samsung")
        val isExynos = hardware.contains("exynos")
        val isSnapdragon = hardware.contains("qcom") || hardware.contains("snapdragon")

        val options = Interpreter.Options().setNumThreads(2)
        var nnApi: NnApiDelegate? = null
        var gpu: GpuDelegate? = null

        if (isSamsung && isExynos) {
            gpu = runCatching { GpuDelegate() }.getOrNull()
            gpu?.let { options.addDelegate(it) }
        } else if (isSamsung && isSnapdragon) {
            nnApi = runCatching { NnApiDelegate() }.getOrNull()
            nnApi?.let { options.addDelegate(it) }
        } else {
            nnApi = runCatching { NnApiDelegate() }.getOrNull()
            nnApi?.let { options.addDelegate(it) }
            if (nnApi == null) {
                gpu = runCatching { GpuDelegate() }.getOrNull()
                gpu?.let { options.addDelegate(it) }
            }
        }

        return DelegateBundle(options = options, nnApiDelegate = nnApi, gpuDelegate = gpu)
    }

    fun createInterpreter(modelBuffer: ByteBuffer): Pair<Interpreter, DelegateBundle> {
        val bundle = createOptions()
        val interpreter = try {
            Interpreter(modelBuffer, bundle.options)
        } catch (_: IllegalArgumentException) {
            val cpuOptions = Interpreter.Options().setNumThreads(2)
            Interpreter(modelBuffer, cpuOptions)
        }
        return interpreter to bundle
    }
}
