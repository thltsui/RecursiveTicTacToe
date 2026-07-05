import re

with open('substack/essay1b_rl_background.md', 'r') as f:
    text = f.read()

# First, remove bold **
text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

# Next, remove italics *
# The pattern means:
# Preceded by start of string, space, or punctuation like (, —, etc.
# Starts with an asterisk
# Contains no asterisks inside (or we can just use non-greedy)
# Ends with an asterisk
# Followed by end of string, space, or punctuation.
text = re.sub(r'(^|[\s(—])\*([^*]+)\*([\s.,;:)—]|$)', r'\1\2\3', text)
# Run it twice in case of adjacent ones
text = re.sub(r'(^|[\s(—])\*([^*]+)\*([\s.,;:)—]|$)', r'\1\2\3', text)

with open('substack/essay1b_rl_background.md', 'w') as f:
    f.write(text)

