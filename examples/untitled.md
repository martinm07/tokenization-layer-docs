---
description: >-
  An example of using this package to make a model (with the tokenization layer)
  and train it on the IMDB Reviews Dataset.
---

# IMDB Reviews

### [**Open this in Google Colab**](https://colab.research.google.com/drive/1Cje-mIoK13G5Af1rGdsnQwNPKv44Pqmg?usp=sharing)\*\*\*\*

```bash
!pip install tokenization-layer
```

```python
import tokenization_layer

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import string
```

### Getting & Preparing the Data

Firstly, let's get the data and prepare it. I've the dataset on my Google Drive, so we can download it from there:

```python
import requests
from io import StringIO
```

```python
orig_url = 'https://drive.google.com/file/d/1-4wZ3VawRfxvX9taPhfHWU7mAiH-gBDe/view?usp=sharing'
file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
data = pd.read_csv(csv_raw)
```

Then, some basic preprocessing:

```python
data = data.sample(len(data)).reset_index(drop=True)

# Strip "<br />" tags and convert to lowercase
data["review"] = data["review"].apply(lambda x: x.replace("<br />", " ").lower())
# Strip punctuation
data["review"] = data["review"].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))
# Get top 30 most common characters
chars = "".join(pd.Series(list(" ".join(data["review"].to_list()))).value_counts().keys()[:30])
# Remove everything except the top 30 most common characters
data["review"] = data["review"].apply(lambda x: re.sub(f"[^{chars}]", "", x))

from sklearn.preprocessing import OrdinalEncoder
data[["sentiment"]] = OrdinalEncoder().fit_transform(data[["sentiment"]])

print(data.head())
```

Do a train-test-validation split:

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_val, y_train, y_val = train_test_split(data["review"], data["sentiment"], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
```

And finally, turn it into a TensorFlow dataset for the final preprocessing; splitting the text by letter, one-hot encoding it and padding it, so that they're all the same length \(+ batch it into 32s\):

```python
one_hot = lambda x: tf.cast(tokenization_layer.one_hot_str(x, chars), tf.float32)
# Clip and pad all reviews to be 2000 characters long
def clip_and_pad(x):
    output_length = 2000
    shape = tf.shape(x)
    if shape[1] >= output_length:
        return x[:, :output_length]
    else:
        return tf.concat([x, tf.zeros((shape[0], output_length-shape[1]))], axis=1)

# Convert to TF Datasets and preprocess
X_train, X_val, X_test = tf.data.Dataset.from_tensor_slices(X_train), tf.data.Dataset.from_tensor_slices(X_val), tf.data.Dataset.from_tensor_slices(X_test)
X_train, X_val, X_test = X_train.map(one_hot).map(clip_and_pad), X_val.map(one_hot).map(clip_and_pad), X_test.map(one_hot).map(clip_and_pad)

y_train, y_val, y_test = tf.data.Dataset.from_tensor_slices(np.asarray(y_train).astype('float32')), tf.data.Dataset.from_tensor_slices(np.asarray(y_val).astype('float32')), tf.data.Dataset.from_tensor_slices(np.asarray(y_test).astype('float32'))

# Merge X and ys into one TF Dataset
train_set, val_set, test_set = tf.data.Dataset.zip((X_train, y_train)), tf.data.Dataset.zip((X_val, y_val)), tf.data.Dataset.zip((X_test, y_test))
for item in train_set.take(3):
    print(item)

# Shuffle, batch and prefetch data
train_set = train_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)
val_set = val_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)
test_set = test_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)

# Add extra dimension so that the shape is `(batch_size, num_chars, text_len 1)` (i.e. what the tokenization layer wants):
train_set = train_set.map(lambda x, y: (tf.expand_dims(x, 3), y))
val_set = val_set.map(lambda x, y: (tf.expand_dims(x, 3), y))
test_set = test_set.map(lambda x, y: (tf.expand_dims(x, 3), y))
```

### Defining the Model

Now, let's start making the model

 To start, we need an initialization method for the patterns \(tokens\) of the tokenization layer. Here we'll use `tokenization_layer.PatternsInitializerMaxCover`.

```python
corpus = " ".join(data["review"].to_list())[:10000000] # If the corpus is too large, we'll run into RAM issues
patterns_init = tokenization_layer.PatternsInitilizerMaxCover(corpus, chars)
```

And then, we'll define our model \(we'll use the subclassing API, but the other keras APIs also work\):

```python
class ModelTokenization(tf.keras.Model):
    def __init__(self):
        super(ModelTokenization, self).__init__(name='')
        
        self.tokenization = tokenization_layer.TokenizationLayer(n_neurons=500, initializer=patterns_init, 
                                                                 pattern_lens=max(patterns_init.gram_lens))
        # Process the output of the tokenization layer so that it's digestible to the Embedding layer
        self.lambda1 = keras.layers.Lambda(lambda x: tf.transpose(tf.squeeze(x, 3), [0, 2, 1]))
        # We only need an embedding length of 1 because the rest of the network is just fully connected...
        self.embedding = tokenization_layer.EmbeddingLayer(embedding_length=1)
        # Flatten the embedded text so that the dense layers can process it
        self.flatten = keras.layers.Flatten()

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.dense = keras.layers.Dense(64)
        self.out = keras.layers.Dense(1, activation="sigmoid")

    def call(self, input_tensor, return_intermediates=False, training=False):
        tokenization_out = self.tokenization(input_tensor, training=training)
        lambda1_out = self.lambda1(tokenization_out, training=training)
        embedding_out = self.embedding(lambda1_out, training=training)
        flatten_out = self.flatten(embedding_out)

        batch_norm1_out = self.batch_norm1(flatten_out, training=training)
        dense_out = self.dense(batch_norm1_out, training=training)
        out = self.out(dense_out, training=training)

        if return_intermediates:
            return out, dense_out, flatten_out, embedding_out, lambda1_out, tokenization_out
        else:
            return out
```

```python
model = ModelTokenization()
_ = model(tf.zeros([32, 31, 2000, 1]))
model.summary()
```

### Making the Training Loop

 For the final thing in this example, we'll make a custom training loop for our model. Note you don't _have_ to do this, `model.compile()` `model.fit()` also works.

```python
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy()
train_acc_metric = keras.metrics.Accuracy()
val_acc_metric = keras.metrics.Accuracy()
```

Our training loop will save model checkpoints, as well as info on how the patterns and gradients evolved. We initialize those things here:

```python
path = "model_checkpoints/"
```

```python
with open(path+"patterns_log.txt", "a+") as f:
    pass
with open(path+"grads_log.csv", "a+") as f:
    f.write("Out Mean,Out Std,Dense Mean,Dense Std,Embedding Mean,Embedding Std,Tokenization Mean,"+\
            "Tokenization Std,Out Kernel Mean,Out Kernel Std,Out Bias Mean,Out Bias Std,"+\
            "Dense Kernel Mean,Dense Kernel Std,Dense Bias Mean,Dense Bias Std,"+\
            "Embedding Kernel Mean,Embedding Kernel Std,Patterns Mean,Patterns Std,\n")
with open(path+"vals_log.csv", "a+") as f:
    f.write("Out Mean,Out Std,Dense Mean,Dense Std,Embedding Mean,Embedding Std,Tokenization Mean,"+\
            "Tokenization Std,Out Kernel Mean,Out Kernel Std,Out Bias Mean,Out Bias Std,"+\
            "Dense Kernel Mean,Dense Kernel Std,Dense Bias Mean,Dense Bias Std,"+\
            "Embedding Kernel Mean,Embedding Kernel Std,Patterns Mean,Patterns Std,\n")
```

Lastly, here's the acutal training loop iself:

```python
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Note that these functions won't work if your `chars` includes and `"<UNK>"`.
char_lookup = tf.concat([tf.constant(["█"]), tf.strings.bytes_split(tf.constant(chars))], axis=0)
reverse_text = lambda x: tf.strings.join(tf.gather(char_lookup, tf.argmax(tf.concat([tf.fill([1, x.shape[1]], 0.5), x], axis=0), axis=0)))
```

```python
epochs = 10
for epoch in range(epochs):
    train_loss_rounded, train_acc_rounded = 0, 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_set):
        # -=-= COMPUTE GRADIENTS OF BATCH  =-=-
        with tf.GradientTape() as tape:
            z, dense_out, flatten_out, embedding_out, lambda1_out, tokenization_out = model(x_batch_train, return_intermediates=True, training=True)
            z = tf.squeeze(z, 1)
            loss = loss_fn(y_batch_train, z)
        layer_vals = [z, dense_out, embedding_out, tokenization_out]
        grads = tape.gradient(loss, layer_vals+model.trainable_variables)
        layer_grads = grads[:len(layer_vals)]
        grads = grads[len(layer_vals):]
        
        # -=-= LOG INGO  =-=-
        progress_bar_done = "".join(["█" for _ in range(round( step*20/len(train_set) ))])
        progress_bar_left = "".join([" " for _ in range(20-round( step*20/len(train_set) ))])
        percent_done = round(step*100/len(train_set), 2)

        save_patterns = False
        if step%10 == 0:
            save_patterns = True
            # Decode patterns
            patterns = model.tokenization.patterns
            patterns = tf.cast(tf.math.logical_and(
                patterns == tf.expand_dims(tf.reduce_max(patterns, axis=0), 0),
                tf.reduce_sum(patterns, axis=0) > 0
            ), tf.float32)
            patterns_decoded = [reverse_text(pattern).numpy().decode() for pattern in tf.transpose(tf.squeeze(patterns, 2), [2, 0, 1])]
            # Get patterns to log
            pattern_grads = tf.transpose(tf.squeeze(grads[0], 2), [2, 0, 1])
            pattern_grads_summary = tf.math.reduce_std(pattern_grads, axis=[1, 2])+tf.abs(tf.reduce_mean(pattern_grads, axis=[1, 2]))
            pattern_grads_sorted_indexes = list(pd.Series(pattern_grads_summary).sort_values().keys())

        clear_output(wait=True)
        print(f'Epoch {epoch+1}/{epochs} - |{progress_bar_done}{progress_bar_left}| - {percent_done}% - {step+1}/{len(train_set)}')
        print(f'Train loss: {train_loss_rounded} - Train accuracy: {train_acc_rounded}')
        print()
        # Log patterns
        top_n = 15
        buffer = "".join("0" for _ in range(7))

        patterns_log_high = [f'"{patterns_decoded[i]}": '+(str(pattern_grads_summary[i].numpy()*100)+buffer)[:7]+" | " 
                             for i in pattern_grads_sorted_indexes[-top_n:]]
        num_per_row = int(np.floor(135/len(patterns_log_high[0])))
        print(f"{color.BOLD}Patterns with diverse non-zero gradients{color.END}")
        for i in range(int(np.floor(len(patterns_log_high)/num_per_row))):
            print("".join(patterns_log_high[(i)*num_per_row:(i+1)*num_per_row]))
        if len(patterns_log_high)%num_per_row != 0:
            print("".join(patterns_log_high[-(int(np.floor(len(patterns_log_high)/num_per_row))*num_per_row)+1:]))

        patterns_log_low = [f'"{patterns_decoded[i]}": '+(str(pattern_grads_summary[i].numpy()*100)+buffer)[:7]+" | " 
                             for i in pattern_grads_sorted_indexes[:top_n]]
        num_per_row = int(np.floor(135/len(patterns_log_low[0])))
        print(f"{color.BOLD}Patterns with mostly zero gradients{color.END}")
        for i in range(int(np.floor(len(patterns_log_low)/num_per_row))):
            print("".join(patterns_log_low[(i)*num_per_row:(i+1)*num_per_row]))
        if len(patterns_log_low)%num_per_row != 0:
            print("".join(patterns_log_low[-(int(np.floor(len(patterns_log_low)/num_per_row))*num_per_row)+1:]))

        # -=-= UPDATE NETWORK & METRICS  =-=-
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(y_batch_train, tf.round(z))
        train_loss_rounded, train_acc_rounded = "%.4f" % loss.numpy(), "%.4f" % train_acc_metric.result().numpy()

        # -=-= SAVE THINGS  =-=-
        if (step%int(np.floor(len(train_set)/5))==0) and (step != 0):
            cp_num = len(os.listdir(path))-1
            model_cp = tf.train.Checkpoint(model=model)
            model_cp.write(path+f"model_cp_{cp_num}/model_checkpoint")
        if save_patterns:
            with open(path+"patterns_log.txt", "a+") as f:
                [f.write(f'"{pattern}", ') for pattern in patterns_decoded]
                f.write("\n")
        with open(path+"grads_log.csv", "a+") as f:
            for layer_index in [0, 1, 2, 3]:
                f.write(str( tf.reduce_mean(layer_grads[layer_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(tf.reduce_mean(layer_grads[layer_index], axis=0)).numpy() )+",")
            for param_index in [6, 7, 4, 5, 1, 0]:
                f.write(str( tf.reduce_mean(grads[param_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(grads[param_index]).numpy() )+",")
            f.write("\n")
        with open(path+"vals_log.csv", "a+") as f:
            for layer_index in [0, 1, 2, 3]:
                f.write(str( tf.reduce_mean(layer_vals[layer_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(tf.reduce_mean(layer_vals[layer_index], axis=0)).numpy() )+",")
            for param_index in [6, 7, 4, 5, 1, 0]:
                f.write(str( tf.reduce_mean(model.trainable_variables[param_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(model.trainable_variables[param_index]).numpy() )+",")
            f.write("\n")

    train_acc_metric.reset_states()
```

 There it is! Every 10 steps, this training loop will decode all the patterns \(i.e. tokens\) in the tokenization layer, save them to `patterns_log.txt`, and display the top patterns with the most non-zero and zero gradients \(which is an indicator of convergance\). It also saves the mean and standard deviation of values and gradients along the whole model, in `grads_log.csv` and `vals_log.csv` respectively.

