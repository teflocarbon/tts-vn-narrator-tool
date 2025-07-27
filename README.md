# TTS Visual Novel Narrator Tool (macOS)

**WARNING: Highly Experimental Proof-of-Concept**

This tool is a very early, unstable, and experimental proof-of-concept for automatically narrating text from Visual Novels on macOS using high-quality TTS (Text-to-Speech) via the OpenAI API. There are hardcoded values for the TTS engine, and the code is subject to change and may break unexpectedly.

## What It Does

- Lets you select a region of your screen containing VN dialogue.
- Uses macOS's built-in screen capture and OCR APIs for maximum performance and reliability.
- Detects when the text in the region changes, extracts the new text, and sends it to a TTS engine (currently hardcoded for OpenAI API).
- Speaks the extracted text aloud, enabling high-quality narration for any visual novel—no VN-side integration required.

## Why Use This?

- Enjoy high-quality TTS narration in your favorite visual novels, even if they don't natively support it.
- No need to modify the VN or use unreliable hacks—this works externally via screen capture and OCR.
- macOS-native: leverages system APIs for best performance and OCR accuracy.

## Limitations & Warnings

- **Very experimental and unstable**. Expect bugs, crashes, and rough edges.
- **Hardcoded values** for TTS engine/API key—edit the code to configure.
- Only tested on macOS; will not work on Windows or Linux.
- Requires Python 3.12 and dependencies listed in `requirements.txt`.
- Not designed for continuous unattended use.

## Usage Instructions

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the tool:**
   ```sh
   python3.12 main.py --interval 1.0
   ```
   (You can adjust the interval in seconds; lower is faster but may use more CPU/cause bugs)

3. **Select the region** of your VN where dialogue appears when prompted.

4. **Listen!** The tool will automatically OCR and narrate new text as it appears. Magic!

## Recommendations

- **Maximize your VN's text speed** (instant/fast-forward), and set auto-advance if possible. This ensures the tool can keep up and minimizes repeated/partial reads.
- If your VN has animated or flashing UI, you may need to tweak the interval or code for best results. macOS's OCR is fairly reliable, but some characters may be missed or misread.

## Configuration

- The TTS engine is currently hardcoded for OpenAI API. You must edit `tts_engine.py` to supply your API key and adjust settings.
- Further customization (voice, language, etc.) may require code changes.

## Disclaimer

This is a personal project and is not affiliated with any visual novel publisher. Use at your own risk. Contributions and feedback are welcome, but stability and support are not guaranteed. 

Be aware of Terms of Service for the TTS API you are using as some content in visual novels may be disallowed.

## Why macOS?

That's what I use and it made OCR and Screen Capture easy. I can't be bothered to make it cross platform.