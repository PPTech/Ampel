# Version: 0.9.5
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT

## ICameraProvider

```kotlin
interface ICameraProvider {
    fun start()
    fun stop()
    fun nextFrame(): ByteArray?
}
```
