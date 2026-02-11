# LLM Text Preprocessing Foundations (Embeddings)

Workshop based on Chapter 2 of *Build a Large Language Model From Scratch* by Sebastian Raschka.

## Key Concepts

### Tokenization

Tokenization is the process of breaking raw text into smaller units (tokens) that a neural network can process. An LLM cannot work with raw strings directly; it needs numbers. We explored two approaches:

- **Simple tokenization** using regex to split by whitespace and punctuation.
- **Byte Pair Encoding (BPE)** via `tiktoken`, which is what GPT-2 actually uses. BPE is smarter because it splits unknown words into subword pieces, so the model never encounters a truly "unknown" token. This is critical for real-world applications where the input can be anything (code, new words, other languages).

### Sliding Window & Data Sampling

LLMs learn by predicting the next token. To prepare training data, we use a sliding window that creates input-target pairs where the target is the input shifted by one position. Two key parameters control this:

- **`max_length`**: the number of tokens per sample (context window size).
- **`stride`**: how many tokens we move the window forward.

When `stride < max_length`, the windows overlap, generating more training samples from the same text. This is useful with small datasets but can lead to overfitting if the overlap is too large.

### Embeddings

An embedding turns a token ID (just a number) into a dense vector. The key insight is that these vectors are learned during training; the neural network adjusts them through backpropagation so that tokens used in similar contexts end up with similar vectors. This is how embeddings "encode meaning."

In neural network terms, the embedding layer is equivalent to one-hot encoding followed by a matrix multiplication, but implemented as a simple lookup table for efficiency. The weights of this layer are trainable, just like any other NN layer.

### Positional Embeddings

Token embeddings alone have no sense of order; the word "cat" gets the same vector regardless of its position. To fix this, GPT-2 adds absolute positional embeddings: a separate learned vector for each position in the context window. The final input to the model is `token_embedding + positional_embedding`.

## Experiment: Effect of `max_length` and `stride`

We tested several combinations of `max_length` and `stride` on the same text (~5,000 tokens) and observed the following:

- **No overlap** (`stride = max_length`): produces the minimum number of samples. For `max_length=4, stride=4` we got ~1,274 samples; for `max_length=256, stride=256` only ~19.
- **With overlap** (`stride < max_length`): dramatically increases the sample count. For `max_length=4, stride=1` we got ~5,097 samples — about **4x more** than without overlap.
- **Bigger context** (`max_length=256`): each sample is longer so the model sees more context at once, but we get far fewer total samples.
- **50% overlap** (`stride = max_length / 2`): roughly doubles the number of samples compared to no overlap. This is a common practical choice that balances data augmentation with the risk of overfitting.

### Conclusions

1. **Overlap is a trade-off**: more overlap = more training data, but the samples are very similar to each other, which can cause the model to memorize instead of generalize.
2. **Context size matters**: a larger `max_length` lets the model learn longer-range dependencies, but requires more memory and yields fewer samples from the same corpus.
3. **Embeddings are the bridge** between discrete tokens and continuous math that neural networks can work with. Without them, there is no way for the model to learn relationships between words.
4. **Positional information is essential**: without it, "the cat sat on the mat" and "the mat sat on the cat" would look identical to the model.
5. **BPE solves the unknown word problem**: unlike simple tokenizers that crash on unseen words, BPE breaks them into known subword pieces, making the model robust to any input.

## References

- Raschka, S. *Build a Large Language Model From Scratch*. Manning, 2024. 