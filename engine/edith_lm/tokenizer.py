"""E.D.I.T.H. V13 Byte-Level BPE Tokenizer with Guaranteed Zero <unk> Fallback.

This tokenizer is completely custom, self-contained, and written without external
proprietary or HuggingFace dependencies.

Key Features:
1. Byte-level fallback covering all 256 individual byte values (0-255).
   It is mathematically IMPOSSIBLE to emit an `<unk>` token for any valid or invalid UTF-8 string.
2. Contraction, punctuation, and number-aware regex pre-tokenization.
3. Special tokens for conversational turn-taking, affective mood, thinking chains,
   and structured action proposals (<user>, <edith>, <think>, <act>, <mood>, <mem>).
4. Full serialization and fast load/save format in JSON.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union


class EdithByteBPETokenizer:
    # Reserved Special Tokens
    SPECIAL_TOKENS = [
        "<pad>",     # 0: Padding
        "<bos>",     # 1: Beginning of sequence
        "<eos>",     # 2: End of sequence
        "<unk>",     # 3: Unused reserved fallback
        "<user>",    # 4: User turn prefix
        "<edith>",   # 5: Assistant turn prefix
        "<think>",   # 6: Start of internal reasoning scratchpad
        "</think>",  # 7: End of internal reasoning scratchpad
        "<act>",     # 8: Start of structured tool action proposal
        "</act>",    # 9: End of structured tool action proposal
        "<mood>",    # 10: Start of affect/mood conditioning
        "</mood>",   # 11: End of affect/mood conditioning
        "<mem>",     # 12: Start of memory fact extraction
        "</mem>",    # 13: End of memory fact extraction
    ]

    # Pre-tokenization regex: splits text into chunks preserving contractions, numbers, punctuation, spaces
    PAT_REGEX = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+""",
        re.UNICODE,
    )


    def __init__(self, vocab_file: Optional[str] = None):
        self.special_tokens = list(self.SPECIAL_TOKENS)
        self.special_token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special_token = {idx: tok for idx, tok in enumerate(self.special_tokens)}
        
        # Byte encoder & decoder (maps byte 0..255 to unique unicode chars to avoid control char issues)
        self.byte_to_char, self.char_to_byte = self._bytes_to_unicode()

        # Vocabulary & Merges
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        self._init_base_vocab()

        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def _bytes_to_unicode(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Maps all 256 bytes to printable, distinct Unicode characters."""
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = [chr(b) for b in bs]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(chr(256 + n))
                n += 1
        d = dict(zip(bs, cs))
        inv_d = {v: k for k, v in d.items()}
        return d, inv_d

    def _init_base_vocab(self):
        """Initializes vocabulary with special tokens followed by all 256 byte tokens."""
        self.encoder = {}
        self.decoder = {}
        
        # 1. Add special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.encoder[tok] = idx
            self.decoder[idx] = tok

        # 2. Add individual byte representations
        offset = len(self.special_tokens)
        for b in range(256):
            ch = self.byte_to_char[b]
            token_id = offset + b
            self.encoder[ch] = token_id
            self.decoder[token_id] = ch

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.special_token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.special_token_to_id["<unk>"]

    # -------------------------------------------------------------------------
    # Training Algorithm (Pair Frequency Counting & Merging)
    # -------------------------------------------------------------------------

    def train_from_corpus(self, texts: List[str], target_vocab_size: int = 16384, verbose: bool = True):
        """Learns Byte-Level BPE merges from a collection of raw text strings."""
        if target_vocab_size < len(self.special_tokens) + 256:
            raise ValueError(f"target_vocab_size must be >= {len(self.special_tokens) + 256}")

        self._init_base_vocab()
        self.merges = []
        self.bpe_ranks = {}

        if verbose:
            print(f"[Tokenizer] Training Byte-BPE on {len(texts)} documents. Target Vocab: {target_vocab_size}...")

        # 1. Pre-tokenize and count word frequencies
        words_list = []
        word_freqs_list = []
        raw_word_counts = {}
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                raw_word_counts[word] = raw_word_counts.get(word, 0) + 1

        for word, freq in raw_word_counts.items():
            word_bytes = [self.byte_to_char[b] for b in word.encode("utf-8")]
            words_list.append(word_bytes)
            word_freqs_list.append(freq)

        num_merges = target_vocab_size - len(self.encoder)
        if verbose:
            print(f"[Tokenizer] Unique byte-level words: {len(words_list)}. Performing up to {num_merges} merges...")

        # 2. Build initial pair counts and inverted index
        pair_counts: Dict[Tuple[str, str], int] = {}
        pair_to_words: Dict[Tuple[str, str], Set[int]] = {}

        for w_idx, word in enumerate(words_list):
            freq = word_freqs_list[w_idx]
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(w_idx)

        # 3. Iteratively merge most frequent adjacent byte-pairs
        for i in range(num_merges):
            if not pair_counts:
                if verbose:
                    print(f"[Tokenizer] Early convergence at merge {i}. Vocab size: {len(self.encoder)}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                if verbose:
                    print(f"[Tokenizer] Merges reached frequency < 2 at iteration {i}. Final Vocab: {len(self.encoder)}.")
                break

            merged_token = best_pair[0] + best_pair[1]
            token_id = len(self.encoder)
            self.encoder[merged_token] = token_id
            self.decoder[token_id] = merged_token
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = len(self.merges) - 1

            affected_w_indices = list(pair_to_words.get(best_pair, set()))
            del pair_counts[best_pair]
            if best_pair in pair_to_words:
                del pair_to_words[best_pair]

            first, second = best_pair
            for w_idx in affected_w_indices:
                word = words_list[w_idx]
                freq = word_freqs_list[w_idx]

                # Remove all old pairs for this word from index
                for k in range(len(word) - 1):
                    p = (word[k], word[k + 1])
                    if p in pair_counts:
                        pair_counts[p] -= freq
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                    if p in pair_to_words:
                        pair_to_words[p].discard(w_idx)

                # Merge occurrences in word
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                        new_word.append(merged_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                words_list[w_idx] = new_word

                # Add new pairs for this word back into index
                for k in range(len(new_word) - 1):
                    p = (new_word[k], new_word[k + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + freq
                    if p not in pair_to_words:
                        pair_to_words[p] = set()
                    pair_to_words[p].add(w_idx)

            if verbose and (i + 1) % 500 == 0:
                print(f"[Tokenizer] Merge {i+1}/{num_merges}: {repr(best_pair[0])} + {repr(best_pair[1])} -> Vocab: {len(self.encoder)}")

        if verbose:
            print(f"[Tokenizer] Training complete. Total vocabulary size: {self.vocab_size}")

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        new_word = []
        i = 0
        first, second = pair
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    # -------------------------------------------------------------------------
    # Encoding & Decoding
    # -------------------------------------------------------------------------

    def _pre_tokenize(self, text: str) -> List[str]:
        # Splits text while preserving punctuation and spacing
        chunks = self.PAT_REGEX.findall(text)
        return chunks if chunks else [text]


    def _bpe(self, token_bytes: Tuple[str, ...]) -> List[str]:
        word = list(token_bytes)
        if len(word) <= 1:
            return word

        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            # Find the pair with lowest merge rank
            best_pair = None
            min_rank = float("inf")
            for p in pairs:
                rank = self.bpe_ranks.get(p, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = p

            if best_pair is None or min_rank == float("inf"):
                break

            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes any text string to token IDs with guaranteed zero <unk> fallback."""
        if not text:
            ids = []
            if add_bos:
                ids.insert(0, self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return ids

        # Split around special tokens first
        special_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        parts = re.split(special_pattern, text)

        token_ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[part])
            else:
                words = self._pre_tokenize(part)
                for word in words:
                    # Convert to byte chars
                    word_bytes = tuple(self.byte_to_char[b] for b in word.encode("utf-8"))
                    bpe_tokens = self._bpe(word_bytes)
                    for tok in bpe_tokens:
                        token_ids.append(self.encoder[tok])

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        byte_list: List[int] = []
        res_strings: List[str] = []

        def flush_bytes():
            nonlocal byte_list
            if byte_list:
                try:
                    res_strings.append(bytes(byte_list).decode("utf-8", errors="replace"))
                except Exception:
                    res_strings.append("")
                byte_list = []

        for tid in token_ids:
            if tid in self.id_to_special_token:
                flush_bytes()
                if not skip_special_tokens:
                    res_strings.append(self.id_to_special_token[tid])
            elif tid in self.decoder:
                char_str = self.decoder[tid]
                for ch in char_str:
                    byte_list.append(self.char_to_byte[ch])
            else:
                # Unknown token ID (should never happen)
                flush_bytes()
                if not skip_special_tokens:
                    res_strings.append("<unk>")

        flush_bytes()
        return "".join(res_strings)

    # -------------------------------------------------------------------------
    # Serialization (JSON format)
    # -------------------------------------------------------------------------

    def save(self, filepath: str):
        """Saves vocabulary and merge ranks to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "version": "13.0",
            "type": "byte_level_bpe",
            "special_tokens": self.special_tokens,
            "vocab": self.encoder,
            "merges": [f"{p[0]} {p[1]}" for p in self.merges],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Loads vocabulary and merge ranks from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_tokens = data.get("special_tokens", list(self.SPECIAL_TOKENS))
        self.special_token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special_token = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        self.encoder = {k: int(v) for k, v in data["vocab"].items()}
        self.decoder = {int(v): k for k, v in data["vocab"].items()}

        self.merges = []
        self.bpe_ranks = {}
        for idx, merge_str in enumerate(data.get("merges", [])):
            parts = merge_str.split(" ", 1)
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                self.merges.append(pair)
                self.bpe_ranks[pair] = idx


def build_default_edith_tokenizer(corpus_texts: Optional[List[str]] = None, save_path: Optional[str] = None) -> EdithByteBPETokenizer:
    """Convenience factory: builds and trains or loads the default E.D.I.T.H. tokenizer."""
    tok = EdithByteBPETokenizer()
    if save_path and os.path.exists(save_path):
        tok.load(save_path)
        return tok

    if corpus_texts:
        tok.train_from_corpus(corpus_texts, target_vocab_size=16384)
        if save_path:
            tok.save(save_path)
    return tok
