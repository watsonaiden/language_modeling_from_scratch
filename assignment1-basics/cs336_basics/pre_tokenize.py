import regex as re
from collections import defaultdict
from typing import NamedTuple


class PretokenDuplicates(NamedTuple):
    tokens: list
    occurences: int


class PretokenizeInputs(NamedTuple):
    file_path: str
    start: int
    end: int
    special_tokens: list[str]


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# prefer single arg for use with subprocess imap which cannot pass multiple args
def subprocess_pre_tokenize_chunk(i: PretokenizeInputs):
    return pre_tokenize(read_chunk(i.file_path, i.start, i.end), i.special_tokens)


def read_chunk(file_path, start, end):
    with open(file_path, "rb") as file:
        file.seek(start)
        return file.read(end - start).decode("utf-8", errors="ignore")


# returns list where each entry is a "segment"
# each segment is represented as a list of integers which are byte values representing the string utf-8 encoded
def pre_tokenize(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    unqiue_pretokens = defaultdict(int)

    for split in re.splititer("|".join(re.escape(special_token) for special_token in special_tokens), text):
        for match in re.finditer(PAT, split):
            unqiue_pretokens[match.group(0).encode()] += 1

    return unqiue_pretokens
