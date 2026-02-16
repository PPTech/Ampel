# AmpelAI Mobile Clean Architecture Plan

Version: 0.9.1  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Project Tree (Target Production Layout)

```text
Ampel/
├── ios/
│   ├── App/
│   │   ├── Presentation/
│   │   │   ├── Dashboard/
│   │   │   ├── CarPlay/
│   │   │   └── Settings/
│   │   ├── Domain/
│   │   │   ├── Entities/
│   │   │   ├── UseCases/
│   │   │   └── Repositories/
│   │   ├── Data/
│   │   │   ├── Local/
│   │   │   ├── Sensors/
│   │   │   └── Navigation/
│   │   └── DI/
│   └── Tests/
├── android/
│   ├── app/
│   │   ├── presentation/
│   │   │   ├── dashboard/
│   │   │   ├── auto/
│   │   │   └── settings/
│   │   ├── domain/
│   │   │   ├── entity/
│   │   │   ├── usecase/
│   │   │   └── repository/
│   │   ├── data/
│   │   │   ├── local/
│   │   │   ├── sensors/
│   │   │   └── navigation/
│   │   └── di/
│   └── tests/
├── ai_engine/
│   ├── training/
│   ├── inference/
│   ├── privacy/
│   └── datasets/
├── features/
│   ├── traffic_ai_agent.feature
│   └── traffic_light_core.feature
└── docs/
    └── MOBILE_CLEAN_ARCHITECTURE.md
```

## Architecture Rules

1. **Maps/Navigation**
   - Mobile map and navigation integration must use **Mapbox Navigation SDK** logic for CarPlay/Android Auto-compatible overlays.
2. **Privacy-by-design**
   - No raw video upload.
   - Face/license-plate blur in edge preprocessing before inference.
   - Keep telemetry minimal, pseudonymized, and configurable retention.
3. **Language Boundaries**
   - iOS app: Swift.
   - Android app: Kotlin.
   - AI model training and offline data prep only: Python.
4. **Clean Architecture**
   - Presentation → Domain → Data dependencies only.
   - Domain layer isolated from framework-specific code.
   - Mapbox/camera/OS APIs isolated behind repository or gateway interfaces.

## Core Domain Use Cases

- Detect traffic light state by lane context.
- Trigger speeding alert if red light + speed over threshold.
- Trigger "Go" alert for green light + stationary timeout.
- Ask user to select correct signal for ambiguous lane-light mapping.
- Apply privacy filters (blur faces/plates) before model processing.
