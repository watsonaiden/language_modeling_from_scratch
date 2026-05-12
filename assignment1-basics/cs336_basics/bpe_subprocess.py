from collections import defaultdict
from typing import NamedTuple
import regex as re


from multiprocessing.connection import Connection


class BytePair(NamedTuple):
    # represent the paired bytes
    first_byte: int
    second_byte: int


STOP_COMMAND = "STOP"
DIFF_COMMAND = "UPDATE"



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
            raise Exception('unknown command', command)


def read_chunk(file_path, start, end):
    with open(file_path, "rb") as file:
        file.seek(start)
        return file.read(end - start).decode("utf-8", errors="ignore")

# returns list where each entry is a "segment"
# each segment is represented as a list of integers which are byte values representing the string utf-8 encoded
def pre_tokenize(text: str, special_tokens: list[str]) -> list[list[int]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return [
        list(match.group(0).encode())
        for split in re.splititer('|'.join(re.escape(special_token) for special_token in special_tokens), text)
        for match in re.finditer(PAT, split)
    ]
    

# return bytePair -> count mapping and bytePair -> segment index locations
def initialize_statistics(pretokens: list[list[int]]) -> tuple[dict[BytePair, int], dict[BytePair, set[int]]]:
    stats = defaultdict(int)
    indexs = defaultdict(set)

    for seg_ind, segment in enumerate(pretokens):
        for first, second in zip(segment, segment[1:]):
            byte_pair = BytePair(first, second)
            stats[byte_pair] += 1
            indexs[byte_pair].add(seg_ind)

    return stats, indexs


def apply_merges(merge_pair: BytePair, new_token: int, index_data: dict[BytePair, set[int]], pretokens: list[list[int]]):
    merged_pair_indexs = index_data.get(merge_pair, [])
    diff = defaultdict(int)
    for segment_index in merged_pair_indexs:
        segment_tokens = pretokens[segment_index]
        ind = 0

        while ind < len(segment_tokens) - 1:
            if tuple(segment_tokens[ind : ind + 2]) == merge_pair:
                segment_tokens[ind : ind + 2] = [new_token]
                diff[merge_pair] -= 1

                # ensure one token infront of pair
                if ind > 0:

                    diff[(segment_tokens[ind - 1], merge_pair[0])] -= 1


                    new_pair = BytePair(segment_tokens[ind - 1], new_token)
                    diff[new_pair] += 1
                    index_data[new_pair].add(segment_index)


                # no element behind our replacement if we are at last index
                if ind < len(segment_tokens) - 1:
                    diff[(merge_pair[1], segment_tokens[ind + 1])] -= 1

                    new_pair = BytePair(new_token, segment_tokens[ind + 1])
                    diff[new_pair] += 1
                    index_data[new_pair].add(segment_index)

            ind += 1

    return diff