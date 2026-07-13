#!/bin/bash
# Generates essay_p3_alphazero.pdf
# Run from repo root OR from substack/: bash substack/generate_pdf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="file://${SCRIPT_DIR}/essay_p3_alphazero.html"
OUTPUT="${SCRIPT_DIR}/essay_p3_alphazero.pdf"

echo "Target: $OUTPUT"

# ── Option 1: copy the already-downloaded jsPDF version from Downloads ──
DOWNLOADS_PDF=~/Downloads/essay_p3_alphazero.pdf
if [ -f "$DOWNLOADS_PDF" ]; then
    cp "$DOWNLOADS_PDF" "$OUTPUT"
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "✅  PDF copied from Downloads: $OUTPUT ($SIZE)"
    exit 0
fi

# ── Option 2: Chrome/Chromium headless (full CSS rendering) ──
CHROME=""
for candidate in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"; do
    if [ -f "$candidate" ]; then CHROME="$candidate"; break; fi
done

if [ -n "$CHROME" ]; then
    echo "Rendering via: $CHROME"
    "$CHROME" --headless --disable-gpu --no-sandbox \
        --print-to-pdf="$OUTPUT" --print-to-pdf-no-header \
        "$INPUT" 2>/dev/null
    if [ -f "$OUTPUT" ]; then
        SIZE=$(du -h "$OUTPUT" | cut -f1)
        echo "✅  PDF saved: $OUTPUT ($SIZE)"
        exit 0
    fi
fi

echo "❌  Could not generate PDF. Open essay_p3_alphazero.html in Chrome and print to PDF manually."
exit 1
