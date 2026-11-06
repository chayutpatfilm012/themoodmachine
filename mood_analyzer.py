# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the sentiment of the word right after them.
NEGATION_WORDS = {"not", "never"}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        tokens: List[str] = []
        for raw_token in cleaned.split():
            # Strip punctuation from the edges only, so "happy!" -> "happy"
            # while inner apostrophes survive ("can't" stays "can't").
            token = raw_token.strip(string.punctuation)

            # A token made only of punctuation is likely an emoticon like ":)"
            # or ":-(", so keep it as-is instead of dropping it.
            tokens.append(token if token else raw_token)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0

        for i, token in enumerate(tokens):
            if token in self.positive_words:
                word_score = 1
            elif token in self.negative_words:
                word_score = -1
            else:
                continue

            # Modeling improvement: negation. If the previous token negates
            # this word, flip its score ("not happy" -> -1, "never bad" -> +1).
            if i > 0 and self.is_negation(tokens[i - 1]):
                word_score = -word_score

            score += word_score

        return score

    def is_negation(self, token: str) -> bool:
        """
        Return True if the token negates the word that follows it.

        Covers "not", "never", and "n't" contractions like "don't" or "can't".
        """
        return token in NEGATION_WORDS or token.endswith("n't")

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        # Find out whether the text has both positive and negative signals.
        # Each word's contribution is how much the score changes when that
        # word is added, so negation is handled exactly as in score_text().
        tokens = self.preprocess(text)
        has_positive = False
        has_negative = False
        previous_score = 0
        for i in range(len(tokens)):
            current_score = self.score_text(" ".join(tokens[: i + 1]))
            if current_score > previous_score:
                has_positive = True
            elif current_score < previous_score:
                has_negative = True
            previous_score = current_score

        if has_positive and has_negative:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
