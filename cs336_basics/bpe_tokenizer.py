from collections.abc import Iterable, Iterator
from functools import lru_cache
from io import BytesIO
import regex as re
import json
import logging
from cs336_basics.bpe_train import CPU_NUM, PAT
from tests.common import gp2_unicode_to_bytes

log = logging.getLogger(__name__)

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.id_2_token = vocab
        self.token_2_id = {token: _id for _id, token in vocab.items()}
        self.merges = merges
        self.pattern = PAT
        self.special_tokens = special_tokens
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.spliter = "|".join([re.escape(t) for t in self.special_tokens])
            for token in special_tokens:
                t = token.encode('utf-8')
                if t not in self.token_2_id:
                    _id = len(self.token_2_id)
                    self.token_2_id[t] = _id
                    self.id_2_token[_id] = t
        log.debug(f"tokenizer pattern {self.pattern}")

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        m = gp2_unicode_to_bytes()
        with open(vocab_filepath, "rb") as f:
            token_2_id = json.load(f)
            id_2_token = {}
            for token, _id in token_2_id.items():
                id_2_token[_id] = bytes(m[c] for c in token)
        with open(merges_filepath, "r") as f:
            merges = [tuple(bytes(m[c] for c in s) for s in line.strip().split(" ")) for line in f.readlines()]
        return cls(id_2_token, merges, special_tokens)
            

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        res = []
        # 1. handle special tokens and remove them before pre-tokenization
        texts = [text]
        if self.special_tokens is not None:
            texts = re.split(f'({self.spliter})', text)
        log.debug(f"split texts {texts}")
        # 2. pre-tokenize
        for text in texts:
            if self.special_tokens is not None and text in self.special_tokens:
                res.append(self.token_2_id[text.encode('utf-8')])
                continue
            for word in re.finditer(self.pattern, text):
                # 3. apply the merges in order
                # 4. encode them into IDs by vocab
                res.extend([self.token_2_id[token] for token in self.merge(word.group())])
        log.debug(f"encode {text} into {res}")
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs
        """
        for text in iterable:
            texts = [text]
            if self.special_tokens is not None:
                texts = re.split(f'({self.spliter})', text)
            for text in texts:
                if self.special_tokens is not None and text in self.special_tokens:
                    yield self.token_2_id[text.encode('utf-8')]
                    continue
                for word in re.finditer(self.pattern, text):
                    for token in self.merge(word.group()):
                        yield self.token_2_id[token]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        buf = BytesIO()
        for id in ids:
            buf.write(self.id_2_token[id])
        return buf.getvalue().decode('utf-8', errors='replace')

    @lru_cache
    def merge(self, word: str) -> list[bytes]:
        """
        apply all token pair merges in order
        """
        # TODO: can optimize merge instead of iterating all merge in order?
        b = word.encode("utf-8")
        before = [b[i : i + 1] for i in range(len(b))]
        for merge in self.merges:
            after = []
            i = 0
            while i + 1 < len(before):
                if (before[i], before[i + 1]) == merge:
                    after.append(before[i] + before[i + 1])
                    i += 2
                else:
                    after.append(before[i])
                    i += 1
            if i == len(before) - 1:
                after.append(before[i])
            before = after
        # log.debug(f"word: {word}, tokens: {before}")
        return before
