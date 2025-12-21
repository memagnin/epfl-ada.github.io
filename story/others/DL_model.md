---
layout: page
title: DL model
subtitle: "Fun Facts"
---

For the [experimental setup]({{ '/story/3_a' | relative_url }}), we setup a deep learning model as follows:

We chose a classic page rank architecture : a context encoder and a candidate encoder that consist of one linear layer, one activation then one linear layer. For each candidate its score is defined by the dot product of the encoded context and the encoded candidate. This model is quite simple but sufficient for this task. All the textual features were encoded using a pretrained BERT transformer.


                                [WikiNavModel]
                                      |
                  +-------------------+-------------------+
                  |                                       |
          [context_encoder]                       [candidate_encoder]
                  |                                       |
        (Input Size: 1536)                      (Input Size: 1540*)
                  |                                       |
                  v                                       v
    +---------------------------+           +---------------------------+
    | Linear (1536 -> 512)      |           | Linear (1540 -> 512)      |
    | Bias: True                |           | Bias: True                |
    +---------------------------+           +---------------------------+
                  |                                       |
                  v                                       v
               [ReLU]                                  [ReLU]
                  |                                       |
                  v                                       v
    +---------------------------+           +---------------------------+
    | Linear (512 -> 256)       |           | Linear (512 -> 256)       |
    | Bias: True                |           | Bias: True                |
    +---------------------------+           +---------------------------+
                  |                                       |
                  v                                       v
         (Output Size: 256)                      (Output Size: 256)
