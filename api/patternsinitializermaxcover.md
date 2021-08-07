---
description: >-
  Keras initializer that uses a corpus of text to initialize the patterns as
  randomly chosen grams from the corpus, weighted by how common said grams are
  respectively.
---

# PatternsInitializerMaxCover

```text
tokenization_layer.PatternsInitilizerMaxCover(
    text_corpus, chars, 
    gram_lens=[5, 6, 7, 8, 9, 10, 11, 12, 13], 
    filter_over=1
)
```

### Parameters

`text_corpus` ---- `str`  
The corpus of text that will be used to generate the patterns. WARNING: You may run into memory issues if the corpus is too big.

`chars` ---- `str`  
String of the one-hot encoding categories \(i.e. characters\) for how the text will be encoded. The index of a character in the string will be said character's index in the encoding. So, for example, `chars = "abcdefghijklmnopqrstuvwxyz"`, then each character in a pattern will be 26-dimensional vector \(i.e. a one-hot encoding with 26 characters\). You may also include `"<UNK>"` in your `chars` \(and it'll be all characters that can't be found in `chars`\). If you don't, unidentified characters will be encoded as not having a category \(i.e. the same as padding\). Finally, _**MAKE SURE THAT THE SAME `CHARS` ARE USED FOR ENCODING ALL TEXT THAT WILL BE INPUT DATA TO NEURAL NET.**_

`gram_lens` ---- `list`, optional \(default: `[5, 6, 7, 8, 9, 10, 11, 12, 13]`\)  
A list of the possible lengths a pattern can be \(patterns will be padded with 0s at the end to be the same length\).

`filter_over` ---- `int`, optional \(default: `1`\)  
The minimum number of time a gram must occur in the corpus to be a possible pattern.

### Example

```python
import re
import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg

corpus = gutenberg.raw("austen-emma.txt")
# Remove arbritray strings of "\\n"s and " "s
corpus = re.sub(r"[\\n ]+", " ", corpus.lower())

chars = "".join(pd.Series(list(corpus)).value_counts(sort=True).keys()) + "<UNK>"
init = tokenization_layer.PatternsInitilizerMaxCover(corpus, chars)

# Initialize patterns of shape `(num_chars, max_len, 1, num_neurons)`
#   Where there are `num_neurons` patterns (one for each neuron), each 
#   with random length/number of characters (but padded to be `max_len`)
#   and each character being a one-hot encoding with `num_chars` 
#   categories.
patterns = init((len(init.chars), max(init.gram_lens), 1, 200))
```

