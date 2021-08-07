---
description: >-
  Keras layer that takes in discrete values/categories (one-hot encoded) and
  embeds (maps) them to continuous values. This embedding is learnt through
  training.
---

# EmbeddingLayer

```text
tokenization_layer.EmbeddingLayer(
    embedding_length, **kwargs
)
```

Takes in matrix of discrete values \(one-hot encoded\) and embeds them into continuous values, trained like the rest of the network. Shape of `X` should be`(batch_size, sequence_length, onehot_categories)`.  
Note that this is essentially the same as `keras.layers.Embedding`, except that this version doesn't have to be the first layer in the network \(i.e. it has an upstream gradient\), unlike the official keras embedding layer.

### Parameters

`embedding_length` ---- `int`  
How many values \(i.e. dimensions in the vector\) each category should be embedded as. The more values the more complex the _meaning_ behind any embedding can be.

`**kwargs`

