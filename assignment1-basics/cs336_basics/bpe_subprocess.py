from collections import defaultdict
from typing import NamedTuple
import regex as re
from collections import Counter


from multiprocessing.connection import Connection

STOP_COMMAND = "STOP"
DIFF_COMMAND = "UPDATE"


class BytePair(NamedTuple):
    # represent the paired bytes
    first_byte: int
    second_byte: int


class PretokenDuplicates(NamedTuple):
    tokens: list
    occurences: int


class IndexData:
    def __init__(self):
        # byte_pair : {index: num_occurences}
        self.indexs = defaultdict(dict)

    def get_indexs(self, pair: BytePair) -> list[int]:
        return self.indexs.get(pair, {}).keys()

    def remove_occurence(self, pair: BytePair, index):
        index_data = self.indexs[pair]

        if index not in index_data:
            return

        # would not longer exist after deleting
        if self.indexs[pair][index] == 1:
            self.indexs[pair].pop(index)
            return

        self.indexs[pair][index] -= 1

    def add_pair(self, pair: BytePair, index: int):
        index_data = self.indexs[pair]
        if index not in index_data:
            index_data[index] = 1
        else:
            index_data[index] += 1

    def remove_index(self, pair: BytePair):
        self.indexs.pop(pair)


# subprocess to handle a chunk of a file
def train_bpe_subprocess(file_path, start, end, special_tokens: list[str], pipe_end: Connection):
    pretokens = pre_tokenize(read_chunk(file_path, start, end), special_tokens)

    stats, metadata = initialize_statistics(pretokens)

    pipe_end.send(stats)

    while True:
        # await response on what bytes to merge
        command = pipe_end.recv()
        if command[0] == STOP_COMMAND:
            return
        elif command[0] == DIFF_COMMAND:
            merge_bytes, new_token = command[1:]
            diff = apply_merges(merge_bytes, new_token, metadata, pretokens)
            pipe_end.send(diff)
        else:
            raise Exception("unknown command", command)


def read_chunk(file_path, start, end):
    with open(file_path, "rb") as file:
        file.seek(start)
        return file.read(end - start).decode("utf-8", errors="ignore")


# returns list where each entry is a "segment"
# each segment is represented as a list of integers which are byte values representing the string utf-8 encoded
def pre_tokenize(text: str, special_tokens: list[str]) -> list[PretokenDuplicates]:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    unqiue_pretokens = defaultdict(int)

    for split in re.splititer("|".join(re.escape(special_token) for special_token in special_tokens), text):
        for match in re.finditer(PAT, split):
            unqiue_pretokens[match.group(0).encode()] += 1

    return [PretokenDuplicates(list(k), v) for k, v in unqiue_pretokens.items()]


# return bytePair -> count mapping and bytePair -> segment index locations
def initialize_statistics(pretokens: list[PretokenDuplicates]) -> tuple[dict[BytePair, int], IndexData]:
    stats = defaultdict(int)
    indexs = IndexData()

    for seg_ind, (segment, count) in enumerate(pretokens):
        for first, second in zip(segment, segment[1:]):
            byte_pair = BytePair(first, second)
            stats[byte_pair] += 1 * count
            indexs.add_pair(byte_pair, seg_ind)

    return stats, indexs


def apply_merges(merge_pair: BytePair, new_token: int, index_data: IndexData, pretokens: list[PretokenDuplicates]):
    merged_pair_indexs = index_data.get_indexs(merge_pair)
    diff = defaultdict(int)
    for segment_index in merged_pair_indexs:
        segment_tokens, count = pretokens[segment_index]
        ind = 0

        while ind < len(segment_tokens) - 1:
            if tuple(segment_tokens[ind : ind + 2]) == merge_pair:
                segment_tokens[ind : ind + 2] = [new_token]
                diff[merge_pair] -= 1 * count

                # ensure one token infront of pair
                if ind > 0:
                    old_pair = BytePair(segment_tokens[ind - 1], merge_pair[0])
                    diff[old_pair] -= 1 * count
                    index_data.remove_occurence(old_pair, segment_index)

                    new_pair = BytePair(segment_tokens[ind - 1], new_token)
                    diff[new_pair] += 1 * count
                    index_data.add_pair(new_pair, segment_index)

                # no element behind our replacement if we are at last index
                if ind < len(segment_tokens) - 1:
                    old_pair = BytePair(merge_pair[1], segment_tokens[ind + 1])
                    diff[old_pair] -= 1 * count
                    index_data.remove_occurence(old_pair, segment_index)

                    new_pair = BytePair(new_token, segment_tokens[ind + 1])
                    diff[new_pair] += 1 * count
                    index_data.add_pair(new_pair, segment_index)

            ind += 1

    index_data.remove_index(merge_pair)

    return diff
