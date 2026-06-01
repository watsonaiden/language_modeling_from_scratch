import regex as re

from typing import Iterable, Iterator

from cs336_basics.pre_tokenize import PAT


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges, special_tokens: list[str] = None):
        self.vocab = vocab
        self.inverted_vocab = {v: k for k, v in vocab.items()}

        self.special_tokens = special_tokens or []
        self.special_tokens.sort(key=len, reverse=True)
        assert isinstance(self.special_tokens, (type(None), list))
        self.merges = merges

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): ...

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

        pretoken_ids = [self.inverted_vocab[bytes([byte])] for byte in text.encode()]

        for merge_pair in self.merges:
            start, end = 0, 1
            while end < len(pretoken_ids):
                if (self.inverted_vocab[merge_pair[0]], self.inverted_vocab[merge_pair[1]]) == (
                    pretoken_ids[start],
                    pretoken_ids[end],
                ):
                    pretoken_ids[start : end + 1] = [self.inverted_vocab[merge_pair[0] + merge_pair[1]]]

                start += 1
                end += 1

        return pretoken_ids

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
