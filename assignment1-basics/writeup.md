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