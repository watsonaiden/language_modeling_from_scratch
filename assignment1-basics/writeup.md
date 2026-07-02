## Problem (unicode1): Understanding unicode



### (a) what unicode charecter does chr(0) return?
chr(0) return \x00 which is the NUL symbol [link](https://symbl.cc/en/0000/?utm_referrer=https%3A%2F%2Fsymbl.cc%2Fen%2Funicode-table%2F).


### (b) How does this character’s string representation (__repr__()) differ from its printed representation?

The `__repr__` shows it's hex representation while print (and it's `__str__`) representation creates a 0 space representation



### (c) What happens when this charecter occurs in text?
As a zero space charector it is not visible in the final string.



## Problem (Unicode2): Unicode encodings

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings

UTF-8 creates the shortest encoded sequence for ascii chars which the majority of our data is (1 byte vs 2 or 4). 


### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

This code does not take into account multiple byte sequences such as 'あ'(b'\xe3\x81\x82') will result in a decode error.


### (c) Give a two-byte sequence that does not decode to any Unicode character(s).
'\xe0\x80'. Uses leading bits indicating a 3 byte sequence but is only 2 bytes.



## Problem (train_bpe_tinystories):  BPE Training on TinyStories 
a) Runtime is roughly 38 seconds on mac m1 chip. Peak memory is ~9GB (~900MB per process * 10 process) during pre-tokenization but ~71MB during main merge steps. longest token is 15 bytes which has 3 matches ['b accomplishment', b' disappointment',b' responsibility'].

b) The most expensive process is by the far the pretokenization step utilizing regex. This process takes ~35 of the 38 seconds of runtime. This includes the multiprocessing operations and setup which seem to be necessary evils.


## Problem (train_bpe_expts_owt):  BPE Training on OpenWebText

runtie uv run cs336_basics/bpe.py  3834.06s user 51.83s system 777% cpu 8:20.00 total Longest token is a tie between(64, b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82') and b'----------------------------------------------------------------'

The repeating dashes is somewhat reasonable the other is suprising. Further investigation seems to show this is a incorrectly encoded and re-encoded junk string sometimes called [mojibake](https://en.wikipedia.org/wiki/Mojibake) 


## Problem (tokenize_experiments)

a) using tiny stories and OWT on tiny stories results in a similar 4 bytes / token compression. 

b) OWT has a better compression on the other dataset. On tiny_stories it has a 4 bytes / token compression very simlar to tiny stories own tokenizer. Tiny stories tokenizer on OWT suffers decreasing to ~3.2 bytes / token.

c) Thoughtput is roughly 5 mb / s. For a 825GB dataset this would take ~165k seconds or ~46 hours

d) uint16 (16 bit unsigned integer) works as we know the token values will be between 0-32,000 at most because of the vocab size. uint16 is the smallest data type capable of handling this range as it ranges from 0 - 65535



## Problem (Transformer LM resource accounting)


  ## Model Configurations

  | Model               | vocab_size | context_length | num_layers | d_model | num_heads | d_ff |
  |---------------------|-----------:|---------------:|-----------:|--------:|----------:|-----:|
  | gpt_xl_long_context | 50257      | 16384          | 48         | 1600    | 25        | 4288 |
  | gpt_xl              | 50257      | 1024           | 48         | 1600    | 25        | 4288 |
  | gpt_large           | 50257      | 1024           | 36         | 1280    | 20        | 4288 |
  | gpt_medium          | 50257      | 1024           | 24         | 1024    | 16        | 4288 |
  | gpt_small           | 50257      | 1024           | 12         | 768     | 25        | 4288 |

  ## FLOP Breakdown

| model_name | vocab_size | context_length | num_layers | d_model | num_heads | d_ff | swiglu | swiglu_pct | qkv_calc | qkv_calc_pct | attention_alg_flops | attention_linear_flops | attention_linear_pct | attention_flops | final_linear | final_linear_pct | rmsnorm_flops | rmsnorm_flops_pct | total_flops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt_xl_long_context | 50257 | 16384 | 48 | 1600 | 25 | 4288.000000 | 3.237346e+13 | 0.242356 | 12079595520000 | 0.090431 | 82463372083200 | 4026531840000 | 0.030144 | 0.617341 | 2634914201600 | 0.019726 | 131104768 | 9.814828e-07 | 1.335783e+14 |
| gpt_xl | 50257 | 1024 | 48 | 1600 | 25 | 4288.000000 | 2.023341e+12 | 0.575335 | 754974720000 | 0.214676 | 322122547200 | 251658240000 | 0.071559 | 0.091595 | 164682137600 | 0.046827 | 8194048 | 2.329971e-06 | 3.516803e+12 |
| gpt_large | 50257 | 1024 | 36 | 1280 | 20 | 3413.333333 | 9.663746e+11 | 0.544560 | 362387865600 | 0.204209 | 193273528320 | 120795955200 | 0.068070 | 0.108911 | 131745710080 | 0.074240 | 6555648 | 3.694161e-06 | 1.774597e+12 |
| gpt_medium | 50257 | 1024 | 24 | 1024 | 16 | 2730.666667 | 4.123225e+11 | 0.498593 | 154618822656 | 0.186970 | 103079215104 | 51539607552 | 0.062323 | 0.124646 | 105396568064 | 0.127449 | 5244928 | 6.342325e-06 | 8.269724e+11 |
| gpt_small | 50257 | 1024 | 12 | 768 | 12 | 2048.000000 | 1.159683e+11 | 0.397609 | 43486543872 | 0.149098 | 38654705664 | 14495514624 | 0.049699 | 0.132531 | 79047426048 | 0.271022 | 3934208 | 1.348882e-05 | 2.916643e+11 |


As the model grows with more layers and more context the attention mechanism begins to dominate otherwise the FF networks are the main source of flops.