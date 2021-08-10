---
description: >-
  tf.keras layer that should (but doesn't have to) be the first layer in the
  neural network. It's for learning and applying a tokenization         scheme
  on the input text.
---

# TokenizationLayer

Placed as the first layer in a neural network, it takes in text that's split by character and one-hot encoded \(i.e. has shape `(batch_size, num_chars, text_len, 1`\), and "tokenizes" it using the trainable parameter `patterns`.

```text
tokenization_layer.TokenizationLayer(
    n_neurons, initializer, pattern_lens, **kwargs
)
```

### Parameters

`n_neurons` ---- `int`  
Number of neurons to be in the layer.

`initializer` ---- `keras.initializers.Initializer`  
Initializer for patterns.

`pattern_lens` ---- `int`  
Length/Number of characters every pattern will be.

`**kwargs`

### Example

```python
from tensorflow import keras
import re
import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg

corpus = gutenberg.raw("austen-emma.txt")
# Remove arbritray strings of "\\n"s and " "s
corpus = re.sub(r"[\\n ]+", " ", corpus.lower())

# We're assuming we got `chars` when preprocessing the train data
init = tokenization_layer.PatternsInitilizerMaxCover(corpus, chars)

model = keras.Sequential([
    tokenization_layer.TokenizationLayer(500, init, max(init.gram_lens)),
    keras.layers.Lambda(lambda x: tf.transpose(tf.squeeze(x, 3), [0, 2, 1])),
    tokenization_layer.EmbeddingLayer(1),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64),
    keras.layers.Dense(1, activation="sigmoid")
])
# Initialize parameters and shapes by calling on dummy inputs
_ = model(tf.zeros((32, 30, 2000, 1)))
_ = model(tf.zeros((50, 30, 2000, 1)))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_data, epochs=10)
```



