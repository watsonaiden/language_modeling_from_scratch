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