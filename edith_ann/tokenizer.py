"""Enhanced Domain-Specific Subword and Morpheme-Aware Tokenizer for E.D.I.T.H."""

import json
import re
from typing import List, Dict, Tuple, Optional


class EdithTokenizer:
    """
    Subword- and morphology-aware domain tokenizer for E.D.I.T.H.
    Handles conversational natural language, contractions, filepaths, code commands,
    and unknown word decomposition.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab:
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}
        else:
            self.vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
            self.inv_vocab = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}

        # Pattern separating words, filepaths, urls, symbols, punctuation, and contractions
        self.token_pattern = re.compile(
            r"[A-Za-z0-9_:\.\-\\/\~%\^\+\*\=]+|'(?:s|t|m|d|ll|re|ve)|[^\s\w]"
        )

        # Common inflectional suffixes for morphological fallback
        self.suffixes = ["ing", "ed", "ly", "es", "s", "ment", "tion", "ness", "able", "ive"]

    def tokenize_raw(self, text: str) -> List[str]:
        """Split text into normalized token strings with contraction handling."""
        if not text:
            return []
        # Normalize quotes and dashes
        cleaned = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
        tokens = self.token_pattern.findall(cleaned.strip().lower())
        return tokens

    def build_vocab(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_vocab_size: int = 12000,
        seed_words: Optional[List[str]] = None
    ):
        """Build vocabulary from texts plus comprehensive domain seed words."""
        freq: Dict[str, int] = {}
        for text in texts:
            tokens = self.tokenize_raw(text)
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda item: (-item[1], item[0]))

        self.vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        next_id = len(self.SPECIAL_TOKENS)

        # First add seed words
        if seed_words:
            for sw in seed_words:
                sw_clean = sw.strip().lower()
                if sw_clean and sw_clean not in self.vocab:
                    self.vocab[sw_clean] = next_id
                    next_id += 1

        # Add most frequent corpus words
        for token, count in sorted_tokens:
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
                if len(self.vocab) >= max_vocab_size:
                    break

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _resolve_token_id(self, token: str) -> int:
        """Resolve token ID with morphological stemming fallback before returning UNK."""
        if token in self.vocab:
            return self.vocab[token]

        # 1. Try stripping punctuation (e.g. "pass." -> "pass")
        clean_punct = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        if clean_punct and clean_punct in self.vocab:
            return self.vocab[clean_punct]

        # 2. Try morphological stemming (e.g. "starts" -> "start", "relaxing" -> "relax")
        for suf in self.suffixes:
            if token.endswith(suf) and len(token) > len(suf) + 2:
                stem = token[:-len(suf)]
                if stem in self.vocab:
                    return self.vocab[stem]
                # Try adding 'e' back (e.g. "creating" -> "create")
                stem_e = stem + "e"
                if stem_e in self.vocab:
                    return self.vocab[stem_e]

        # 3. Try hyphen or slash split
        if "-" in token or "/" in token or "\\" in token:
            subparts = re.split(r"[-/\\]", token)
            for p in subparts:
                if p in self.vocab:
                    return self.vocab[p]

        return self.UNK_ID

    def encode(
        self,
        text: str,
        max_len: Optional[int] = None,
        add_special_tokens: bool = True,
        pad_to_max: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Encodes a string into token IDs and attention mask.
        Returns:
            (input_ids, attention_mask)
        """
        raw_tokens = self.tokenize_raw(text)
        token_ids = [self._resolve_token_id(t) for t in raw_tokens]

        if add_special_tokens:
            token_ids = [self.BOS_ID] + token_ids + [self.EOS_ID]

        length = len(token_ids)
        if max_len is not None:
            if length > max_len:
                token_ids = token_ids[:max_len]
                if add_special_tokens and max_len > 0:
                    token_ids[-1] = self.EOS_ID
                attention_mask = [1] * max_len
            else:
                attention_mask = [1] * length
                if pad_to_max:
                    padding = [self.PAD_ID] * (max_len - length)
                    token_ids.extend(padding)
                    attention_mask.extend([0] * len(padding))
        else:
            attention_mask = [1] * length

        return token_ids, attention_mask

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to a readable string."""
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in [self.PAD_ID, self.BOS_ID, self.EOS_ID]:
                continue
            token_str = self.inv_vocab.get(tid, self.UNK_TOKEN)
            tokens.append(token_str)
        return " ".join(tokens)

    def save(self, filepath: str):
        """Serialize tokenizer vocabulary to disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "EdithTokenizer":
        """Load tokenizer vocabulary from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
