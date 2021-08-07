---
description: Initialize patterns using a gaussian/normal distribution.
---

# PatternsInitializerGaussian

```text
tokenization_layer.PatternsInitializerGaussian(
    mean=0.5, stddev=0.15
)
```

### Parameters

`mean` ---- `float`, optional \(default: `0.5`\)  
Mean of gaussian/normal distribution.

`stddev` ---- `float`, optional \(default: `0.15`\)  
Standard deviation of gaussian/normal distribution.

Note that the default values \(`0.5` and `0.15`\) are good for making values between 0 and 1.

