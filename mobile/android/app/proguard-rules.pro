# Version: 0.9.11
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

#############################################
# Ampel app obfuscation policy
#############################################
# Obfuscate/optimize Ampel implementation classes by default.
-keep class com.pptech.ampel.MainActivity { *; }
-keep class com.pptech.ampel.ui.MainScreenKt { *; }

# Keep CameraX classes to prevent reflective/runtime failures.
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# Keep TensorFlow Lite classes and delegates.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# Remove debug/verbose logging to reduce information leakage.
-assumenosideeffects class android.util.Log {
    public static int d(...);
    public static int v(...);
}

# Keep annotations/signatures required by kotlin/runtime tooling.
-keepattributes *Annotation*,Signature,EnclosingMethod,InnerClasses

# Keep encrypted preferences/crypto classes.
-keep class androidx.security.crypto.** { *; }
-dontwarn androidx.security.crypto.**
