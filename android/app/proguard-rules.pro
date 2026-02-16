# Version: 0.9.3
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

# Obfuscate core logic classes in release builds.
-keep,allowobfuscation class com.ampelai.detector.TrafficLightDetector { *; }
-keep,allowobfuscation class com.ampelai.agent.AIAgent { *; }

# Keep TensorFlow Lite interfaces required by reflection/native loading.
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.support.** { *; }
-dontwarn org.tensorflow.lite.**

# Keep model metadata assets identifiers only (names may be read at runtime).
-keepclassmembers class ** {
    public static final java.lang.String MODEL_*;
}
