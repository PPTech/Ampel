# Version: 0.9.4
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT

## Migration to `src/ampel_app`

1. Runtime entrypoint now delegates to package CLI:
   - `traffic_ai_assist.py` -> `ampel_app.cli:main`
2. New package layout:
   - `src/ampel_app/core` (models/rules/localization)
   - `src/ampel_app/privacy` (redaction/anonymization)
   - `src/ampel_app/storage` (DB utilities)
   - `src/ampel_app/ai` (detector/lane-filter/agent)
   - `src/ampel_app/server` (template + API helpers)
3. Backward compatibility:
   - Existing runner script commands continue to invoke `traffic_ai_assist.py`.
4. New CLI GDPR actions:
   - `--gdpr-export`
   - `--gdpr-erase --retention-days N`
5. Versioning:
   - Use `scripts/bump_version.sh <semver>` for coordinated version updates.
