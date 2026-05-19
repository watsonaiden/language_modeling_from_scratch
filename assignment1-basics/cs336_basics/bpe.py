import os
from pydantic import BaseModel
from typing import BinaryIO, NamedTuple
from collections import defaultdict, Counter
from tqdm import tqdm

from multiprocessing import cpu_count, Pool

from cs336_basics.pre_tokenize import PretokenizeInputs, subprocess_pre_tokenize_chunk

import heapq


class BPEStats(BaseModel):
    merges: list[tuple[bytes, bytes]]
    vocab: dict[int, bytes]  # token num to bytes


class BytePair(NamedTuple):
    # represent the paired bytes
    first_byte: int
    second_byte: int


# avoids duplicates by counting the number of occurences
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


class TokenComparison:
    def __init__(self, count, bytes1, bytes2):
        self.count = count
        self.byte1 = bytes1
        self.byte2 = bytes2

    # treat like __gt__ to invert structure for max Heap
    def __lt__(self, o: "TokenComparison"):
        if self.count == o.count:
            if self.byte1 == o.byte1:
                return self.byte2 >= o.byte2
            return self.byte1 >= o.byte1
        return self.count >= o.count

    def __eq__(self, o: "TokenComparison"):
        return (self.count == o.count) and (self.byte1 == o.byte1) and (self.byte2 == o.byte2)

    def __repr__(self):
        return f"TokenComparison({self.byte1}+{self.byte2} = {self.count})"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# return bytePair -> count mapping and bytePair -> segment index locations
def init_stats(pretokens: list[PretokenDuplicates]) -> tuple[dict[BytePair, int], IndexData]:
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


def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str]) -> BPEStats:
    with open(file_path, "rb") as f_stream:
        chunk_boundaries = find_chunk_boundaries(f_stream, cpu_count(), special_tokens[0].encode())

    args = [
        PretokenizeInputs(file_path, boundary[0], boundary[1], special_tokens)
        for boundary in zip(chunk_boundaries, chunk_boundaries[1:])
    ]

    pretokens_with_counts = Counter()
    with Pool(cpu_count()) as p:
        for pretokens in p.imap_unordered(subprocess_pre_tokenize_chunk, args):
            pretokens_with_counts.update(pretokens)

    # convert to list for easier indexing and mutation of key after merges
    pretokens = [PretokenDuplicates(list(k), v) for k, v in pretokens_with_counts.items()]
    del pretokens_with_counts
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # token# -> byte value
    merges: list[BytePair] = []

    stats, index_data = init_stats(pretokens)

    # heap to avoid reoccuring max calculations
    # need to use custom construct since max heap does not exist easily in py3.12 introduced in 3.14
    # TokenComparison inverts so if > it will say < and vice versa which will mean min in heap is actually max
    stats_heap = [(TokenComparison(v, vocab.get(k[0]), vocab.get(k[1])), k) for k, v in stats.items()]
    heapq.heapify(stats_heap)

    num_merges = vocab_size - len(vocab) - len(special_tokens)

    for _ in tqdm(range(num_merges)):
        most_common_pair = None
        while not most_common_pair:
            top_pair = heapq.heappop(stats_heap)
            # validate if pair is stale
            # since we do not clean the heap there may be stale entries if a pairs count was decremented.
            # if stats is not equal this entry is old and there is more up to date entry in the heap
            if top_pair[0].count != stats[top_pair[1]]:
                continue

            most_common_pair = top_pair[1]

        token1, token2 = most_common_pair

        new_token = len(vocab)
        merges.append((vocab[token1], vocab[token2]))
        vocab[new_token] = vocab[token1] + vocab[token2]

        merge_diffs = apply_merges(most_common_pair, new_token, index_data, pretokens)
        for pair, change in merge_diffs.items():
            stats[pair] += change
            if stats[pair] <= 0:
                del stats[pair]

            else:
                heapq.heappush(stats_heap, (TokenComparison(stats[pair], vocab.get(pair[0]), vocab.get(pair[1])), pair))

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()

    return BPEStats(merges=merges, vocab=vocab)


if __name__ == "__main__":
    # print(train_bpe('data/simple_file.txt', 256+1+6, special_tokens=['<|endofsentence|>']).merges)
    # print(train_bpe('data/based_example.txt', 256+1+6, special_tokens=['<|endofsentence|>']).merges)
    # train_bpe_master('/Users/awatsy/projects/language_modeling_from_scratch/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt', 500, special_tokens=['<|endoftext|>'])
    bpe = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10_000, special_tokens=["<|endoftext|>"])
    print("longest token", max((len(word), word) for word in bpe.vocab.values()))
    print("last token", max(bpe.vocab.items()))
