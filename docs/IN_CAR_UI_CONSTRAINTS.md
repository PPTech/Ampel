# Version: 0.9.4
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT

## In-Car UI Constraints (CarPlay / Android Auto)

- Head-unit UI must remain template-based and low-distraction.
- MVP must **not** stream raw/live camera video to CarPlay/Android Auto displays.
- Allowed in-car elements:
  - status indicator (red/yellow/green/unknown)
  - short textual alerts
  - audio/haptic warnings
- Any debug video overlays are limited to local phone debug screens only.
