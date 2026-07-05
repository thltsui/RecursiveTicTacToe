#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./substack_export.sh <markdown_file>"
    exit 1
fi

if ! command -v pandoc &> /dev/null; then
    echo "Pandoc not found. Installing via Homebrew..."
    brew install pandoc
fi

echo "Converting $1 to Substack format..."
# Convert Markdown to HTML, then HTML to RTF, and copy to clipboard
pandoc "$1" -f markdown -t html | textutil -stdin -format html -convert rtf -stdout | pbcopy

echo "✅ Success! The formatted text is now on your clipboard."
echo "👉 Go to Substack, open a new draft, and press Cmd+V to paste."
