plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.sankalp"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.sankalp"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0-prototype"

        // Ed25519 via JCA needs API 33+; Tink path works from minSdk.
        // ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // --- CameraX (capture + lifecycle binding) ---
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    implementation("androidx.lifecycle:lifecycle-runtime:2.8.6")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")

    // --- OpenCV Android SDK (strip detection, warp, per-slot analysis) ---
    // Requires the OpenCV Android SDK AAR; add it as a local module or
    // flatDir dependency. Call OpenCVLoader.initLocal() before first use.
    // implementation(project(":openCVLibrary")) // uncomment when SDK added

    // --- TensorFlow Lite (planned learned slot classifier; model NOT shipped) ---
    val tfliteVersion = "2.16.1"
    implementation("org.tensorflow:tensorflow-lite:$tfliteVersion")
    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:$tfliteVersion")
    implementation("org.tensorflow:tensorflow-lite-api:$tfliteVersion")
    // NNAPI delegate ships inside tensorflow-lite itself.

    // --- Google Tink (Ed25519 receipt signing, preferred path) ---
    implementation("com.google.crypto.tink:tink-android:1.13.0")
}
