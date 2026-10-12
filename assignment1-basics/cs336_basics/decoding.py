import torch

from cs336_basics.modules import TransformerLM, softmax
from cs336_basics.tokenization.tokenizer import Tokenizer


@torch.no_grad()
def generate_next_token(
    model: TransformerLM, prev_tokens: torch.Tensor, temp: float = 1.0, top_p: float = 1.0
) -> torch.Tensor:
    logits = model(prev_tokens)[-1]

    if temp == 0:
        return torch.argmax(logits)

    else:
        probabilities = softmax(logits / temp, dim=-1)

        probabilities, indices = torch.sort(probabilities, descending=True)

        # top_p is smallest set of p that exceeds top_p
        # subtract sorted to get previous sum rather than current
        # if previous sum > top_p we already have a set exceeding so this element not needed
        removed = torch.cumsum(probabilities, dim=0) - probabilities > top_p

        probabilities.masked_fill_(removed, 0)
        probabilities = probabilities / torch.sum(probabilities)

        sampled_idx = torch.multinomial(probabilities, num_samples=1)

        return indices[sampled_idx]


def generate_completion(
    model: TransformerLM, tokenizer: Tokenizer, prompt: str = "", temp=1.0, top_p=1.0, max_tokens=256, stop_tokens=None
) -> str:
    stop_tokens = stop_tokens or ["<|endoftext|>"]

    stop_token_ids = torch.tensor([tokenizer.encode(stop_token) for stop_token in stop_tokens])

    # unsqueeze since first dim is batch so we want 1 batch of size token_count
    text = torch.tensor(tokenizer.encode(prompt))

    max_new_tokens = max_tokens - len(text)

    while max_new_tokens:
        new_token = generate_next_token(model, text, temp, top_p)
        # print(f'generated token {new_token}')
        text = torch.cat((text, new_token))
        if torch.isin(new_token, stop_token_ids):
            break

        max_new_tokens -= 1

    return tokenizer.decode(text.tolist())


if __name__ == "__main__":
    from cs336_basics.checkpoint import get_checkpoint_model
    from cs336_basics.modules.transformer_lm import TransformerLM
    from cs336_basics.tokenization.tokenizer import Tokenizer

    config_path = "/Users/awatsy/projects/language_modeling_from_scratch/assignment1-basics/data/train_config/tiny_story_train.toml"

    lm = get_checkpoint_model(config_path)

    tokenizer = Tokenizer.from_files(
        "data/bpe_trained/tiny_stories_merges.txt",
        "data/bpe_trained/tiny_stories_vocab.json",
        special_tokens=["<|endoftext|>"],
    )

    print(generate_completion(lm, tokenizer, "Once upon a time there was a little boy named Ben."))
