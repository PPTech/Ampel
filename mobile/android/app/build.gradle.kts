/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val releaseSigSha256 = (findProperty("OFFICIAL_SIG_SHA256") as String?)
    ?: System.getenv("OFFICIAL_SIG_SHA256")

android {
    namespace = "com.pptech.ampel"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.pptech.ampel"
        minSdk = 26
        targetSdk = 34
        versionCode = 920
        versionName = "0.9.20"

        // Debug value is non-blocking; release value is enforced below.
        buildConfigField("String", "OFFICIAL_SIG_SHA256", "\"DEBUG_BYPASS\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )

            if (releaseSigSha256.isNullOrBlank() || releaseSigSha256 == "REPLACE_WITH_RELEASE_CERT_SHA256") {
                throw GradleException(
                    "OFFICIAL_SIG_SHA256 must be provided for release builds via -POFFICIAL_SIG_SHA256 or environment variable",
                )
            }
            buildConfigField("String", "OFFICIAL_SIG_SHA256", "\"${releaseSigSha256}\"")
        }
        debug {
            isMinifyEnabled = false
            buildConfigField("String", "OFFICIAL_SIG_SHA256", "\"DEBUG_BYPASS\"")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    val cameraxVersion = "1.3.4"
    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")

    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.compose.material3:material3:1.2.1")

    implementation("com.google.accompanist:accompanist-permissions:0.35.1-alpha")

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    testImplementation("junit:junit:4.13.2")
}
