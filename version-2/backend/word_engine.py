"""
SignBridge v2.0 — Word Engine
Letter buffer → autocomplete → sentence builder
"""

import time
from spellchecker import SpellChecker

# Common ASL fingerspelled words for autocomplete seed
COMMON_WORDS = [
    "hello", "help", "water", "food", "yes", "no", "please", "thank",
    "sorry", "good", "bad", "name", "what", "where", "when", "how",
    "need", "want", "like", "love", "work", "home", "school", "family",
    "friend", "time", "day", "night", "today", "tomorrow", "morning",
    "eat", "drink", "sleep", "walk", "come", "stop", "wait", "more",
    "done", "ready", "nice", "fine", "okay", "sure", "right", "left",
    "here", "there", "this", "that", "with", "from", "have", "been",
]

SPACE_SIGN   = "space"
DELETE_SIGN  = "del"
NOTHING_SIGN = "nothing"

LETTER_PAUSE = 2.0   # seconds of no input before auto-space


class WordEngine:

    def __init__(self):
        self.spell         = SpellChecker()
        self.spell.word_frequency.load_words(COMMON_WORDS)

        self.current_word  = ""
        self.sentence      = []
        self.last_letter_t = None
        self.session_log   = []   # list of (timestamp, letter/word, type)

    # ── LETTER INPUT ─────────────────────────────────────────

    def push_letter(self, letter: str) -> dict:
        """
        Called when a letter is confirmed (hold-to-confirm fired).
        Returns current state.
        """
        letter = letter.upper()
        now    = time.time()

        # Auto-space if long pause
        if self.last_letter_t and (now - self.last_letter_t) > LETTER_PAUSE:
            if self.current_word:
                self._commit_word()

        self.last_letter_t = now

        if letter == SPACE_SIGN.upper() or letter == " ":
            self._commit_word()
        elif letter == DELETE_SIGN.upper():
            self._delete()
        elif letter == NOTHING_SIGN.upper():
            pass
        elif letter.isalpha() and len(letter) == 1:
            self.current_word += letter
            self.session_log.append({
                "ts": now, "value": letter, "type": "letter"
            })

        return self.get_state()

    def _commit_word(self):
        if not self.current_word:
            return
        word = self.current_word.lower()
        self.sentence.append(word)
        self.session_log.append({
            "ts": time.time(), "value": word, "type": "word"
        })
        self.current_word = ""

    def _delete(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence.pop()

    # ── AUTOCOMPLETE ─────────────────────────────────────────

    def get_suggestions(self, prefix: str, n: int = 4) -> list[str]:
        """Return up to n word suggestions for the current prefix."""
        if not prefix:
            return []
        prefix = prefix.lower()
        # Exact prefix match from known words
        candidates = [
            w for w in self.spell.word_frequency.words()
            if w.startswith(prefix) and len(w) > len(prefix)
        ]
        # Sort by frequency
        candidates.sort(
            key=lambda w: self.spell.word_frequency[w], reverse=True
        )
        return candidates[:n]

    def accept_suggestion(self, word: str) -> dict:
        """User tapped an autocomplete suggestion."""
        self.current_word = ""
        self.sentence.append(word.lower())
        self.session_log.append({
            "ts": time.time(), "value": word, "type": "suggestion"
        })
        return self.get_state()

    # ── STATE ────────────────────────────────────────────────

    def get_state(self) -> dict:
        prefix      = self.current_word
        suggestions = self.get_suggestions(prefix)
        sentence    = " ".join(self.sentence)
        if prefix:
            sentence = (sentence + " " + prefix).strip()

        return {
            "current_word" : self.current_word,
            "sentence"     : sentence,
            "word_count"   : len(self.sentence),
            "letter_count" : sum(len(w) for w in self.sentence) + len(self.current_word),
            "suggestions"  : suggestions,
        }

    def clear_word(self) -> dict:
        self.current_word = ""
        return self.get_state()

    def clear_sentence(self) -> dict:
        self.current_word = ""
        self.sentence     = []
        return self.get_state()

    def get_session_log(self) -> list:
        return self.session_log

    def get_sentence_text(self) -> str:
        parts = list(self.sentence)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)
