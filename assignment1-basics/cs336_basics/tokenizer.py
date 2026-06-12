from collections import defaultdict
import heapq
import regex as re

from typing import Iterable, Iterator

from cs336_basics.pre_tokenize import PAT
from cs336_basics.bpe import BPEStats

from dataclasses import dataclass


@dataclass
class Token:
    # represent the paired bytes
    next_token: "Token" = None
    prev_token: "Token" = None
    token: bytes = b""

    def merge_next(self: "Token") -> "Token":
        t2 = self.next_token
        if t2 is None:
            raise RuntimeError("called merge on end token", self)
        # merge into one token
        self.token = self.token + t2.token
        self.next_token = t2.next_token
        if t2.next_token is not None:
            t2.next_token.prev_token = self

        # wipe t2 so it's obvious its no longer active
        # Without this it is hard to tell if a node is dangling since we may still reference due to lazy checking
        t2.token, t2.next_token, t2.prev_token = b"", None, None

        return self


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges, special_tokens: list[str] = None):
        self.vocab = vocab
        self.inverted_vocab = {v: k for k, v in vocab.items()}

        self.special_tokens = special_tokens or []
        self.special_tokens.sort(key=len, reverse=True)
        assert isinstance(self.special_tokens, (type(None), list))

        self.merges = merges
        self.merges_priority = {merge: count for count, merge in enumerate(merges)}

    @classmethod
    def from_files(cls, merges_file_path, vocab_file_path, special_tokens=None):
        bpe = BPEStats.from_files(merges_file_path, vocab_file_path)
        return cls(bpe.vocab, bpe.merges, special_tokens=special_tokens)

    def regex_split(self, text: str) -> Iterator[str]:
        if self.special_tokens:
            splits = re.splititer(
                f"({'|'.join(f'{re.escape(special_token)}' for special_token in self.special_tokens)})", text
            )
        else:
            splits = [text]

        for split in splits:
            # don't want to further subtokenize these are a special case
            if split in self.special_tokens:
                yield split

            else:
                for match in re.finditer(PAT, split):
                    yield match.group(0)

    def encode(self, text: str) -> list[int]:
        encodings = []

        for split in self.regex_split(text):
            encodings.extend(self._tokenize_single_pre_token(split))

        return encodings

    def _tokenize_single_pre_token(self, text: str) -> list[int]:

        # special case we can't break down special tokens they will always become a single token
        if text in self.special_tokens:
            return [self.inverted_vocab[text.encode()]]

        # fake starter token for easier iteration
        fake_starter = Token()
        prev = fake_starter

        for byte_value in text.encode():
            t = Token(token=bytes([byte_value]), prev_token=prev)
            prev.next_token = t
            prev = t

        start = fake_starter.next_token

        pairs = defaultdict(list)
        heap = []
        t1, t2 = start, start.next_token
        while t2 is not None:
            pair = (t1.token, t2.token)
            pairs[pair].append(t1)
            if pair in self.merges_priority:
                heap.append((self.merges_priority[pair]))

            t1 = t1.next_token
            t2 = t2.next_token

        # heap contains merge ids with earliest merge at the top
        heapq.heapify(heap)
        while heap:
            top = heapq.heappop(heap)
            expected_bytes = self.merges[top]

            for first_token in pairs[expected_bytes]:
                if first_token.next_token is None:
                    continue
                elif (first_token.token, first_token.next_token.token) != expected_bytes:
                    continue

                new_token = first_token.merge_next()

                if new_token.prev_token is not None:
                    pair = (new_token.prev_token.token, new_token.token)
                    if pair in self.merges_priority:
                        heapq.heappush(heap, self.merges_priority[pair])
                        pairs[pair].append(new_token.prev_token)

                if new_token.next_token is not None:
                    pair = (new_token.token, new_token.next_token.token)
                    if pair in self.merges_priority:
                        heapq.heappush(heap, self.merges_priority[pair])
                        pairs[pair].append(new_token)

        output = []
        # first "token" is just used as an anchor
        fake_starter = fake_starter.next_token
        while fake_starter:
            output.append(self.inverted_vocab[fake_starter.token])
            fake_starter = fake_starter.next_token

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
         cannot naively encode each str indepdentently as the final part could be split on a boundary
         for example imagine the file 'hi there \n\n\n llm's are cool'
         if you open it and iterate through it you will get
        'hi there \n'
        '\n'
        '\n'
        " llm's are cool"
         if we greedily encode the \n will be split, if we encoded all at once the \n would all be grouped together
        """

        prev = ""

        encoding = []

        for part in iterable:
            split = self.regex_split(prev + part)

            prev = next(split)

            # we can't outright get the final token from a generator so do a lagging processing
            for data in split:
                encoding.extend(self.encode(prev))
                prev = data

        # flush trailing buffer once iterable is done
        if prev:
            encoding.extend(self.encode(prev))

        return encoding

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token] for token in ids).decode(errors="replace")
