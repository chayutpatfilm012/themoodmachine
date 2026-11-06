"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "fire",
    "hyped",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "trash",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "I absolutely love getting stuck in traffic for two hours 🙄",
    "no cap this new album is fire 🔥",
    "Not bad at all, honestly",
    "Got the job offer but I have to move away from all my friends 🥲",
    "Wow, another Monday. Can't wait 😒",
    "Bus comes at 8:15 tomorrow",
    "lowkey stressed about finals but highkey excited for summer",
    "That exam was so easy I'm literally dead 💀😂",
    "Finally submitted my thesis and I can't stop smiling 😭🎉",
    "Ugh, flight got cancelled and nobody at the desk can help us",
    "Heading to the airport at 6, back next Thursday",
    "Proud of how we played but so sad we lost in overtime",
    "This new update is mid tbh, nothing works like before",
    "Coffee with my best friend after months apart ☕ needed this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "negative",  # "I absolutely love getting stuck in traffic for two hours 🙄"
    "positive",  # "no cap this new album is fire 🔥"
    "positive",  # "Not bad at all, honestly"
    "mixed",     # "Got the job offer but I have to move away from all my friends 🥲"
    "negative",  # "Wow, another Monday. Can't wait 😒"
    "neutral",   # "Bus comes at 8:15 tomorrow"
    "mixed",     # "lowkey stressed about finals but highkey excited for summer"
    "positive",  # "That exam was so easy I'm literally dead 💀😂"
    "positive",  # "Finally submitted my thesis and I can't stop smiling 😭🎉"
    "negative",  # "Ugh, flight got cancelled and nobody at the desk can help us"
    "neutral",   # "Heading to the airport at 6, back next Thursday"
    "mixed",     # "Proud of how we played but so sad we lost in overtime"
    "negative",  # "This new update is mid tbh, nothing works like before"
    "positive",  # "Coffee with my best friend after months apart ☕ needed this"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
