# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

This card covers both versions and compares them.

## 1. Model Overview

**Model type:**  
I compared both models. The rule based model is the main version; the ML model was used as a point of comparison.

**Intended purpose:**  
Classify short, informal text posts (similar to social media updates) into one of four moods: `positive`, `negative`, `neutral`, or `mixed`. It is an educational project for exploring how simple classifiers succeed and fail, not a production tool.

**How it works (brief):**  
- *Rule based:* The text is lowercased and split into words, and punctuation at the edges of each word is removed. Each word found in a positive word list adds +1 to a score and each word in a negative word list subtracts 1. If a word is directly preceded by a negation ("not", "never", or a word ending in "n't"), its value is flipped. If the post contains both positive and negative signals, it is labeled `mixed`; otherwise a score above 0 is `positive`, below 0 is `negative`, and 0 is `neutral`.
- *ML:* Each post is turned into word counts (bag of words), and a logistic regression model learns which words are associated with each label from the labeled examples.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` contains 20 posts with 20 matching labels in `TRUE_LABELS`. The dataset was built in three stages:

1. 6 starter posts provided with the assignment.
2. 8 additional posts that added slang, emojis, sarcasm, and mixed feelings (14 total).
3. 6 more posts with new vocabulary, drafted by an AI assistant, added for a final round of testing (20 total).

Final label counts: positive (7), negative (6), mixed (4), neutral (3).

**Labeling process:**  
Labels are specified manually in `TRUE_LABELS`, one per post; neither model generates them. They come from three sources: the 6 starter posts were labeled in the assignment template, the 8 second-stage posts were added with their own labels, and the final 6 posts were drafted together with their labels by an AI assistant. Each label is intended to capture the mood the writer most likely meant, not the mood suggested by individual words. For example, "I absolutely love getting stuck in traffic for two hours 🙄" is labeled `negative` because it is sarcastic, even though it contains "love". `mixed` was used when a post clearly expressed two opposing feelings, such as "Proud of how we played but so sad we lost in overtime".

Several posts were hard to label and could reasonably receive another label:

- "That exam was so easy I'm literally dead 💀😂" (`positive`) relies on "dead" being slang for laughing.
- "Feeling tired but kind of hopeful" (`mixed`) could be read as mildly negative.
- "This is fine" (`neutral`) is sometimes used sarcastically online to mean things are not fine.
- "Finally submitted my thesis and I can't stop smiling 😭🎉" (`positive`) uses 😭 to mean tears of joy rather than sadness.

**Important characteristics of your dataset:**  

- Contains slang ("no cap", "lowkey", "fire", "mid", "tbh")
- Contains emojis (🙄, 🔥, 🥲, 😒, 💀, 😂, 😭, 🎉, ☕)
- Includes sarcasm ("Wow, another Monday. Can't wait 😒")
- Some posts express mixed feelings
- Contains short or ambiguous messages ("This is fine")

**Possible issues with the dataset:**  
- It is very small (20 posts), so a single post changes accuracy by 5 percentage points.
- `neutral` has only 3 examples, so the labels are somewhat imbalanced.
- Labels come from several sources and reflect individual judgment rather than agreement between multiple labelers; others might disagree on sarcastic or mixed posts.
- The informal language is mostly US internet slang and does not represent other dialects, regions, or languages.
- The rule based word lists were adjusted after looking at failures on the first 14 posts, so those posts are not a fully independent test for that model.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  

- **Preprocessing:** lowercase the text, split on whitespace, and strip punctuation from the start and end of each word, so "happy!" matches "happy" while "can't" stays intact. Tokens made only of punctuation (such as ":)") are kept as-is, and emojis are kept as tokens.
- **Word scores:** +1 for each word in `POSITIVE_WORDS`, -1 for each word in `NEGATIVE_WORDS`. All words have equal weight.
- **Negation (enhancement):** if the previous word is "not", "never", or ends in "n't", the current word's score is reversed. For example, "not happy" scores -1 and "Not bad at all" scores +1.
- **Mixed detection:** if the post contains at least one positive and at least one negative signal (after negation), it is labeled `mixed` regardless of the total score.
- **Thresholds:** otherwise, score > 0 is `positive`, score < 0 is `negative`, and score = 0 is `neutral`.
- **Vocabulary update (enhancement):** "fire" and "hyped" were added to the positive list and "trash" to the negative list to address missed slang.
- **Emoji handling:** emojis are preserved during preprocessing but are **not** scored, because no emojis are in either word list.

**Strengths of this approach:**  
- It is transparent: every prediction can be traced to specific matched words.
- It handles direct, simple statements well, such as "I love this class so much" and "Today was a terrible day".
- Negation handling correctly reverses short phrases like "not happy" and "not bad".
- It behaves consistently and does not need training data.

**Weaknesses of this approach:**  
- Any word outside the 23 listed words scores 0, so most slang, emojis, and less common emotion words ("hopeful", "proud", "terrified") are ignored.
- It reads words literally and cannot detect sarcasm.
- Negation only looks one word back, so "not very happy" is not reversed.
- `mixed` is only detected when both a positive and a negative listed word appear.

**Trade-off from the vocabulary fix:**  
Adding "fire", "hyped", and "trash" fixed the targeted slang sentences ("This movie is fire 🔥", "I'm hyped fr", "This is straight trash") and one original post ("no cap this new album is fire 🔥"). However, because the model matches words without context, it also introduced errors on literal uses: "Take out the trash" became `negative` and "The fire alarm went off" became `positive`, where both should be `neutral`. "My apartment caught fire" was already wrong (`neutral`) and became `positive`, the opposite of its true meaning. The fix moved part of the problem rather than solving it.

## 4. How the ML Model Works (if used)

**Features used:**  
Bag of words using scikit-learn's `CountVectorizer` with default settings. This lowercases text and counts words of two or more letters or digits. It drops punctuation, emojis, and single-letter words, and it ignores word order. The classifier is `LogisticRegression(max_iter=1000)`.

**Training data:**  
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS`. It does not use the rule based word lists.

**Training behavior:**  
Training accuracy was 100% on both the 14-post and the 20-post datasets. Leave-one-out accuracy (training on all posts except one, predicting the held-out post, and repeating for every post) changed very little: 3/14 (21.43%) before and 4/20 (20%) after adding the final 6 posts. The model got the same 3 original posts right in both cases, so the extra data did not noticeably improve generalization.

**Strengths and weaknesses:**  
- *Strength:* it learns patterns from labeled data automatically, with no hand-written rules.
- *Weakness:* with only 20 short posts, it memorizes the training examples instead of learning general patterns. On unseen posts it mostly predicts `positive`, the most common label, and performs worse than simply always guessing `positive` (7/20, 35%).
- *Weakness:* it discards emojis and word order, so it cannot use emoji signals or tell "not happy" apart from other combinations of the same words.

## 5. Evaluation

**How you evaluated the model:**  
Both models were evaluated on the labeled posts in `dataset.py`.

The rule based model's accuracy changed in three steps:

1. 8/14 (57.14%) on the original 14 posts, before the slang vocabulary fix.
2. 9/14 (64.29%) after adding "fire", "hyped", and "trash". Because these words were chosen after looking at failures on these same posts, this result is somewhat optimistic.
3. 10/20 (50%) after adding the final 6 posts, with no further changes to the model.

| Model | 14 posts | 20 posts |
|---|---|---|
| Rule based (after vocabulary fix) | 9/14 (64.29%) | 10/20 (50%) |
| ML, training accuracy | 14/14 (100%) | 20/20 (100%) |
| ML, leave-one-out | 3/14 (21.43%) | 4/20 (20%) |

The rule based model's predictions on the original 14 posts did not change when the final 6 posts were added. Its drop from 64.29% to 50% happened because it got only 1 of the 6 new posts right, not because the model changed.

The ML training accuracy is misleading because the model is tested on the same posts it was trained on, so it has already seen every answer. Leave-one-out is a fairer estimate, but with only 20 posts even that result is too small and noisy to prove how either model would perform on real, unseen text.

**Examples of correct predictions:**  
- "I love this class so much" → `positive`. "love" is in the positive list and nothing negates it.
- "Not bad at all, honestly" → `positive`. "bad" scores -1, but it follows "not", so negation flips it to +1.
- "lowkey stressed about finals but highkey excited for summer" → `mixed`. "stressed" and "excited" are both recognized, so mixed detection applies.

**Examples of incorrect predictions:**  
- "I absolutely love getting stuck in traffic for two hours 🙄" → predicted `positive`, true `negative`. The model counts "love" as +1, ignores 🙄, and has no words for the unpleasant situation.
- "Got the job offer but I have to move away from all my friends 🥲" → predicted `neutral`, true `mixed`. Both feelings are expressed through phrases rather than listed words, so the score is 0.
- "Proud of how we played but so sad we lost in overtime" → predicted `negative`, true `mixed`. "sad" is recognized but "proud" is not.

**How the failures differed:** Both models failed on the sarcastic and mixed posts (for example, the traffic post, "Wow, another Monday. Can't wait 😒", and the job offer post). The ML model also failed in leave-one-out on simple posts the rule based model handled correctly, such as "Today was a terrible day" and "I am not happy about this", because 19 training posts are not enough for it to learn what those words mean.

## 6. Limitations

- **Sarcasm:** "I absolutely love getting stuck in traffic for two hours 🙄" is classified `positive` because the model reads "love" literally. Detecting sarcasm requires understanding the whole sentence, which word-level rules cannot do. I chose to document this as a limitation instead of adding a rule, because a quick test showed that scoring 🙄 as negative only changed the prediction to `mixed`, and any sarcasm rule would risk mislabeling sincere posts.
- **Missing slang:** before the vocabulary fix, "This movie is fire 🔥" and "This is straight trash" were both `neutral`. After the fix, other slang such as "I fw this heavy" and "This new update is mid tbh" is still missed.
- **Ignored emojis:** emojis are kept as tokens but never scored, so strong signals like 🙄, 😒, 🔥, and 💀😂 have no effect.
- **Mixed emotions:** "Feeling tired but kind of hopeful" is predicted `negative` because "hopeful" is not in the word list, so only one side of the feeling is detected.
- **Contextual ambiguity:** the same word can mean opposite things. "fire" is positive in "this album is fire" but negative in "My apartment caught fire", and the model cannot tell the difference.
- **Small dataset:** 20 posts is too few to measure accuracy reliably or to train the ML model well.

## 7. Ethical Considerations

**Overt bias:**  
In a deliberately selected language-style test, I compared four pairs of sentences with the same meaning written in standard English and in informal or slang English. Before the vocabulary fix, the rule based model got all 4 standard English sentences right and all 4 slang sentences wrong (each was labeled `neutral`). Within the model this failure is systematic: the word lists contain only standard English words, so feelings expressed in slang are treated as no feeling at all. Informal language varies across communities, age groups, and regions, so a model with a limited vocabulary may perform unequally across language styles, and people who mostly write informally could have their posts misread more often. After the fix, slang accuracy on this test rose to 3/4, but only for the specific words added.

This test was very small (8 sentences), the pairs were chosen to expose the problem, and the sentences were not linked to any real users or groups. It shows that performance can differ by language style, but it cannot establish bias against any demographic group or measure how large the effect would be on real posts.

**Covert compliance:**  
I used an AI coding assistant to implement the rule based methods, draft the final 6 posts and their labels, run tests, and help write this report. When I asked how to fix missed slang, it proposed adding "fire", "hyped", and "trash", but it also warned that words like "fire" have multiple meanings, predicted that literal uses would be misclassified, and suggested testing counter-examples before trusting the fix. Those regression tests confirmed its warnings. It also pointed out that the ML model's 100% training accuracy was misleading and ran a leave-one-out evaluation, and it noted that the final 6 posts it drafted mostly avoided the listed words, which made them harder for the rule based model. I treated the assistant as a tool for checking results rather than as the final authority: I decided which changes to make and which limitations to document instead of fixing.

**Appropriate use and uncertainty:**  
This model is an educational exercise. It does not report how confident it is, and a `neutral` prediction often means "no listed words found" rather than "no emotion expressed". Its accuracy estimates come from a tiny, self-labeled dataset and are highly uncertain.

**Fairness and real-world risk:**  
A mood classifier used in real applications could misread messages from people whose language differs from the word lists, or miss a message expressing distress written in slang, sarcasm, or emojis. Analyzing personal messages also raises privacy concerns. For these reasons, this model should not be used for consequential decisions such as content moderation, mental health screening, hiring, or grading.

## 8. Ideas for Improvement

- Collect many more labeled posts, ideally labeled by several people, to reduce noise and personal bias in the labels.
- Evaluate on a separate held-out test set that is never used to tune the word lists, instead of reporting training accuracy.
- Add a small emoji lexicon (for example, 🔥 and 😂 as positive, 😒 as negative) and test it for side effects, since emojis like 🙄 often signal sarcasm rather than a simple mood.
- Extend negation to look more than one word back and handle common phrases such as "can't wait".
- Add weighted words so that strong words like "hate" count more than mild ones like "tired".
- For the ML model, try TF-IDF features and keep emojis as tokens, and compare results using leave-one-out or cross-validation.
- Consider a model that uses context, such as a small transformer, to handle sarcasm and words with multiple meanings.
