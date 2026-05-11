from collections import defaultdict
import regex as re
from pydantic import BaseModel


class BPEStats(BaseModel):
    merges: list[tuple[bytes,bytes]]
    vocab: dict[int, bytes] # token num to bytes


def train_bpe(data: str, vocab_size: int, special_tokens: list[str]) -> dict:
    assert vocab_size >= 256+len(special_tokens), 'vocab must atleast cover all bytes and special tokens'
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    merges: list[tuple[bytes,bytes]] = []
    vocab : dict[int, bytes] = {x: bytes([x]) for x in range(256)} # token# -> byte value

    num_merges = vocab_size - 256 - len(special_tokens)

    og_vocab_size = len(vocab)
    for i, special_word in enumerate(special_tokens):
        vocab[og_vocab_size+num_merges+i] = special_word.encode('utf-8')


    seg_tokens: list[list[int]] = []
    for split in re.splititer('|'.join(re.escape(word) for word in special_tokens), data): 
        for match in re.finditer(PAT, split):
            seg_tokens.append(list(match.group(0).encode('utf-8')))

    for i in range(num_merges):
        stats = defaultdict(int)
        for tokens in seg_tokens:
            generate_stats(tokens, stats=stats)
            
        # use the raw bytes for lexigraphic tie breaking rather than token_id
        pair = max(stats.items(), key=lambda x: (x[1], vocab.get(x[0][0]), vocab.get(x[0][1])))[0]

        token1, token2 = pair

        new_token = og_vocab_size + i
        merges.append((vocab[token1], vocab[token2]))
        vocab[new_token] = vocab[token1] + vocab[token2]

        for tokens in seg_tokens:
            apply_merge(tokens, pair, new_token)

    return BPEStats(merges=merges, vocab=vocab)
    
def generate_stats(token_ids: list[int], stats: defaultdict):
    stats = stats

    for pair in zip(token_ids, token_ids[1:]):
        stats[pair] += 1

    return stats



def apply_merge(tokens: list[int], merge_pair: tuple[int, int], new_token: int):

    merge_pair = list(merge_pair)

    ind = 0

    while ind < len(tokens)-1:
        if tokens[ind:ind+2] == merge_pair:
            tokens[ind:ind+2] = [new_token]
        
        ind += 1


    return tokens



