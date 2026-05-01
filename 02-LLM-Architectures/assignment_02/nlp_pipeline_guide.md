# From Tokens to Dinosaurs

**A Field Guide to NLP Pipelines & Char-RNNs — Hometask 2.**

> A study companion for Hometask 2 — covering the AG News classification pipeline (BPE → Word2Vec → MLP) and the character-level LSTM that hallucinates fake dinosaur names. Concepts only — no solutions.

---

## Contents

1. [The pipeline, end to end](#1-the-pipeline-end-to-end)
2. [Why subword tokenization exists](#2-why-subword-tokenization-exists)
3. [BPE — by frequency, greedily](#3-bpe--by-frequency-greedily)
4. [WordPiece — by likelihood](#4-wordpiece--by-likelihood)
5. [Word embeddings — the geometry of meaning](#5-word-embeddings--the-geometry-of-meaning)
6. [Word2Vec: CBOW & Skip-gram](#6-word2vec-cbow--skip-gram)
7. [Choosing dimension & window](#7-choosing-dimension--window)
8. [From tokens to a sentence vector](#8-from-tokens-to-a-sentence-vector)
9. [The MLP — a tiny universal approximator](#9-the-mlp--a-tiny-universal-approximator)
10. [Loss, optimizer, batches, epochs](#10-loss-optimizer-batches-epochs)
11. [Reading a confusion matrix](#11-reading-a-confusion-matrix)
12. [Error analysis & interpretability](#12-error-analysis--interpretability)
13. [Char-level language modeling](#13-char-level-language-modeling)
14. [A PyTorch Dataset for sliding windows](#14-a-pytorch-dataset-for-sliding-windows)
15. [One-hot encoding, slowly](#15-one-hot-encoding-slowly)
16. [LSTMs — why a gate fixes a memory](#16-lstms--why-a-gate-fixes-a-memory)
17. [Generation: greedy, random, top-k](#17-generation-greedy-random-top-k)
18. [Temperature — the variance dial](#18-temperature--the-variance-dial)
19. [The ten tasks, distilled](#19-the-ten-tasks-distilled)

---

## 1. The pipeline, end to end

A neural network does not eat sentences. It eats numbers. The job of an NLP pipeline is to translate "the dollar fell against the yen" into a fixed-size float vector that a classifier can score, while losing as little meaning as possible along the way.

```
text  ──→  tokens  ──→  embeddings  ──→  pooled vec  ──→  MLP  ──→  class
"oil       ["oil",      4 vectors      one vector       d→h→4    "Business"
 rises      "rises",    of size d      of size d         logits
 sharply"   "sharp",
            "ly"]
```

Five transformations, four arrows. Each box is a homework task:

| Arrow | Task | Output |
|-------|------|--------|
| text → tokens | 1.1 | list of subword strings |
| tokens → embeddings | 1.2 | sequence of d-dim vectors |
| embeddings → pooled vec | 1.3 (pooling) | one d-dim vector |
| pooled vec → class | 1.3 (MLP) + 1.4 (eval) | predicted label |

Part 2 is a different shape. Instead of taking text in and emitting a label, the char-RNN takes a string and emits the **next character**. Roll that arrow forward 30 times, feeding each prediction back in, and you get a generated dinosaur name.

> **Every model in this homework is a translator. Text → number → text.**

---

## 2. Why subword tokenization exists

The naive way to split text is by whitespace. The vocabulary becomes the set of unique words seen in training. This works until you meet a word you've never seen — every misspelling, every proper noun, every made-up dinosaur name becomes `[UNK]` and all its meaning is gone. AG News has hundreds of thousands of unique words; storing one weight per word would need a 200,000 × 100 embedding table — twenty million parameters of which most are useless one-time celebrity surnames.

The opposite extreme is character-level: vocabulary of ~30 symbols, no `[UNK]` ever, but each token carries almost no meaning by itself.

**Subword tokenization is the compromise.** Common words stay whole (`the`, `movie`). Rare words get split into recognizable pieces (`unbelievable` → `un` + `believ` + `able`). Truly novel words get split into characters as a fallback. Vocabulary size is fixed (you choose — typically 8k-50k) and you never produce `[UNK]`.

> **Side note — the morphology bonus**
> Once the tokenizer learns the suffix `##able`, every adjective ending in -able shares the embedding for that suffix. The model never had to see `belibably` to suspect it might be an adjective.

### The two algorithms you'll use

| Property | BPE | WordPiece |
|---|---|---|
| Merge criterion | Most frequent adjacent pair | Pair that most increases corpus likelihood |
| Score | $\text{count}(A,B)$ | $\text{count}(A,B) / (\text{count}(A) \cdot \text{count}(B))$ |
| Continuation marker | None — pieces are independent | `##` prefix on non-first pieces |
| Used by | GPT-2/3/4, Llama, RoBERTa | BERT, DistilBERT, Electra |
| Tends to produce | Slightly longer pieces | Slightly shorter, more linguistically motivated pieces |

---

## 3. BPE — by frequency, greedily

Walk through with a tiny toy corpus: `low low low low low lower lower newest newest newest`. Start by splitting every word into characters and counting:

```
l o w           ×5    (was "low")
l o w e r       ×2    (was "lower")
n e w e s t     ×3    (was "newest")
```

Now count every adjacent character pair across the whole corpus:

- `(l, o)` appears 5 + 2 = 7 times
- `(o, w)` appears 5 + 2 = 7 times
- `(e, w)` appears 3 times
- `(w, e)` appears 2 + 3 = 5 times

The most frequent pair (a tie at 7) is something like `(l, o)`. Merge it into `lo`:

```
lo w        ×5
lo w e r    ×2
n e w e s t ×3
vocab adds: "lo"
```

Recount. Now `(lo, w)` appears 7 times — the new most frequent. Merge. Repeat until vocab hits target size. Each merge is a permanent rule. At inference, tokenization just replays those rules in order.

### What this gives you

BPE has no notion of "where in the word am I". Tokens are whole subwords with no marker. The reconstructed sentence is `" ".join(tokens)` after some post-processing — but be careful: original spacing is gone. BPE is also **language-agnostic**; it works on any byte stream, which is why GPT models use a byte-level variant for handling emojis and non-Latin scripts.

> **Side note — Why "byte"**
> Pure character BPE breaks on UTF-8 because a single emoji can be 4 bytes. Byte-level BPE treats input as raw bytes and lets merges cross UTF-8 boundaries, so any unicode sequence is representable.

---

## 4. WordPiece — by likelihood

WordPiece is BPE's more probabilistic cousin. Same merge-pairs procedure, but instead of picking the most-frequent pair, it picks the pair whose merge most increases the likelihood of the corpus. In practice this means scoring each candidate pair `(A, B)` by:

$$\text{score}(A, B) = \frac{\text{count}(A, B)}{\text{count}(A) \cdot \text{count}(B)}$$

This penalizes pairs whose pieces are individually very common. `(t, h)` might appear 100,000 times, but `t` and `h` are each so common that the score is small. `(##ing, well)` might appear only 200 times, but if `##ing` is rare and `well` is rare, the score is large. WordPiece prefers merges that "explain" something — pieces that mostly only appear together.

### The ## marker

WordPiece marks every piece that is *not* the start of a word with `##`. So `unbelievable` tokenizes as `['un', '##bel', '##ie', '##va', '##ble']`. This gives the model an explicit signal: "I am in the middle of a word". The reconstructed sentence is unambiguous — strip `##` and concatenate; if no `##`, add a space.

### Side-by-side example

| Sentence | Word-level | BPE | WordPiece |
|----------|------------|-----|-----------|
| `the unbelievable tokenizers` | `[the, [UNK], [UNK]]` | `[the, un, believ, able, token, izers]` | `[the, un, ##bel, ##iev, ##able, token, ##izers]` |

**Takeaway for the question in Task 1.1:** BPE is simpler, faster to train, and ubiquitous in autoregressive LLMs because it produces clean concatenable pieces. WordPiece's likelihood criterion produces slightly more "morphologically tidy" splits, which is why bidirectional models (BERT) used it. **Both crush word-level tokenization** on rare-word handling: a word-level tokenizer with a 10k vocab will have hundreds of `[UNK]`s on AG News. A subword tokenizer with the same vocab will have zero, because every novel word can be expressed as a sequence of pieces.

---

## 5. Word embeddings — the geometry of meaning

After tokenizing, each token is just an integer ID — say, "movie" is 4137. To do anything useful, the model needs a continuous representation. The naive choice is a one-hot vector of size `vocab_size`: a 10,000-long vector of zeros with a single 1 at index 4137. Two problems: it's huge, and it's dumb. Every pair of words is exactly the same distance apart. "king" is no closer to "queen" than to "asparagus".

An **embedding** replaces each token's one-hot with a dense vector of size `d` — typically 50, 100, or 300. The values are real numbers, learned from data. After training, `king` and `queen` end up close in embedding space, while `king` and `asparagus` are far apart. This is not magic; it's a direct consequence of **the distributional hypothesis**: words that appear in similar contexts tend to have similar meanings.

> **You shall know a word by the company it keeps. — J. R. Firth, 1957.**

### What "close" means

The standard distance for embeddings is **cosine similarity** — the cosine of the angle between two vectors. Magnitude is ignored; only direction matters.

$$\cos(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}$$

Two vectors pointing the same direction have cosine 1; perpendicular ones have cosine 0; opposite ones have cosine −1. `model.wv.most_similar("movie")` just computes cosine similarity between "movie"'s vector and every other vector in the vocab, then returns the top 10.

---

## 6. Word2Vec: CBOW & Skip-gram

Word2Vec is the original embedding-learning algorithm that made "king − man + woman ≈ queen" famous. Two flavors that differ only in **which way the prediction runs**.

### The window

Both flavors slide a window across the corpus. With `window=2`, around the target "fell" in `the dollar fell against the yen`:

```
(the)  [dollar]  ⟨fell⟩  [against]  [the]  (yen)
        ──── context ───  TARGET  ── context ───
```

(`( )` = outside window, `[ ]` = context, `⟨ ⟩` = target)

### CBOW — context predicts word

Continuous Bag-of-Words takes the four context vectors, averages them into one vector, and uses that vector to predict the target word. The training signal is softmax cross-entropy: "given this context, what is the probability over every word in the vocab that the next word is X?"

Because we average four contexts into one prediction, every training sample is one update. **CBOW is fast.** It also smooths over rare words.

### Skip-gram — word predicts context

Skip-gram flips the arrow. Given the word "fell", predict each of the four context words independently. So one occurrence of "fell" generates *four* training samples instead of one. Skip-gram is slower per epoch but extracts more signal from each word occurrence, which makes it shine on rare words and small datasets.

| | CBOW | Skip-gram |
|---|---|---|
| Predicts | target from context | context from target |
| Speed | ~3-5× faster | slower (more updates per token) |
| Best for | large corpora, frequent words | small corpora, rare words |
| Quality on infrequent tokens | poor | strong |
| `sg` flag in gensim | `0` | `1` |

> **Side note — Why this matters for AG News**
> AG News has ~120k news headlines. After BPE tokenization with vocab=10k, you have a decent-sized training set. Both should work, but Skip-gram tends to produce slightly higher-quality embeddings for downstream classification; CBOW will train faster.

---

## 7. Choosing dimension & window

Two free knobs: `vector_size` (embedding dimension) and `window` (context radius). The assignment asks you to sweep `{50, 100, 200}` for the dimension.

### The embedding dimension

Higher dimension = more capacity to distinguish concepts, but also more parameters, more risk of overfitting if data is small, and longer training. There's no universal best:

- `d = 50`: enough for very basic semantic clusters. Cheap. Good when corpus is <1M tokens.
- `d = 100`: the sweet spot for medium corpora (10M-100M tokens). What you'd use by default.
- `d = 200-300`: standard for large corpora (Wikipedia-scale). Often plateaus around 300.

Watch what happens to your downstream accuracy as you sweep. If 200 is barely better than 100, the bottleneck is data, not dimension.

### The window size

The assignment hints `window=5`, the gensim default. Smaller windows (2-3) capture **syntactic** similarity — words with the same role cluster (verbs with verbs, nouns with nouns). Larger windows (8-15) capture **topical** similarity — words about the same general subject. For news classification, somewhere in the middle (5-10) is a good compromise.

### `min_count`, and the actual learned vocab

`min_count=2` means: drop any token that appeared fewer than 2 times. This is the source of one of the questions in Task 1.2 — your tokenizer vocab is 10,000, but your Word2Vec vocab will be smaller, possibly considerably so. Why? Because **the tokenizer has seen every token at least once** (it built the vocab from the corpus), but Word2Vec drops tokens that are too rare to learn anything reliable from. With `min_count=2`, every singleton token is excluded.

> **Side note — The OOV problem you've just created**
> At inference, when you tokenize the test sentence, some BPE tokens may not be in the Word2Vec vocab. This is why the skeleton has `if tok in model.wv:` guards. The fallback (zero vector for empty sentences) is your last line of defense.

---

## 8. From tokens to a sentence vector

After tokenizing, each sentence is a list of token IDs of variable length: 8 tokens for a short headline, 40 for a long one. After embedding, that becomes 8 or 40 vectors of size `d`. But an MLP needs **fixed-size** input. So you have to compress a variable-length sequence of d-dimensional vectors into one d-dimensional vector. This is **pooling**.

| Pool | Formula | What it captures | Failure mode |
|------|---------|------------------|--------------|
| Mean | $\bar{v} = \frac{1}{n}\sum_i v_i$ | Average direction. Robust default. | Long sentences with one strong topic word get diluted. |
| Sum | $s = \sum_i v_i$ | Total mass. Magnitude grows with length. | Sentence length leaks into the vector — model may learn "long = X". |
| Max | $m_j = \max_i v_{i,j}$ | Strongest activation per dimension. | Loses count information ("good good good" same as "good"). |

For news classification, **mean pooling** is the strongest default — it gives every token equal weight, doesn't leak length, and the embedding average usually ends up in the right semantic neighborhood.

### The unspoken assumption

Pooling throws away word order. *"Dog bites man"* and *"Man bites dog"* produce identical pooled vectors. For headline classification this is fine — topic words ("market", "election", "championship") usually carry the signal. For tasks where order matters — sentiment, NLI, translation — pooled bag-of-embeddings is too weak and you'd reach for an RNN, CNN, or Transformer. **This is exactly the limitation Task 1.3 asks you to articulate.**

> **Pooling is to text what averaging your monthly weather is to climate. Useful summary, lost story.**

---

## 9. The MLP — a tiny universal approximator

A Multi-Layer Perceptron is just stacked logistic regressions with non-linearities glued between them. Each layer is a `nn.Linear` (a matrix multiply + bias). Each non-linearity is something like ReLU. Stack two or three and you have, in principle, a function that can approximate anything.

Minimal architecture for AG News:

```
input  : (batch, 100)            ← pooled embedding, dim = 100
linear : 100 → 256
relu
linear : 256 → 4                 ← one logit per class
softmax (inside the loss)
```

### Depth and width

With pooled bag-of-embeddings as input, you've already lost most of the structural information. A deep MLP cannot recover it. **One or two hidden layers is plenty** — past that you mostly memorize. Width is more forgiving: 128, 256, 512 are all reasonable. Rule of thumb: hidden_dim ≈ input_dim, possibly 2× input_dim.

### ReLU and friends

ReLU is $\max(0, x)$ — let positive signals through, zero out negatives. Simple, fast, gradient-friendly. Without non-linearity, a stack of `nn.Linear` layers is mathematically identical to a single `nn.Linear`. You'd just be computing $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$. The non-linearity is the only thing that buys you "deep".

### Dropout, optionally

If you see big gaps between training and test accuracy, add `nn.Dropout(0.3)` between layers. It zeros out 30 % of activations randomly during training, forcing the model not to rely on any single feature. Don't add it preemptively — measure first.

> **Side note — What an MLP cannot do here**
> It cannot model word order. It cannot model long-range dependencies. It treats the sentence as a single point in a 100-d cloud. RNNs, CNNs, and Transformers all process the sequence token by token, preserving structural information that pooling threw away. **This is the limitation Task 1.3 asks you to name.**

---

## 10. Loss, optimizer, batches, epochs

### The loss

AG News has 4 classes. The right loss is **cross-entropy** — multi-class generalization of binary cross-entropy. PyTorch's `nn.CrossEntropyLoss` takes raw logits (no softmax) and integer labels, and does the softmax + log + negative-log-likelihood internally — numerically stable, fast.

$$L = -\frac{1}{N}\sum_i \log\!\left(\frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}}\right)$$

### The optimizer

Adam is the default and rarely wrong. Learning rate `1e-3` is its happy place. If your training loss isn't dropping after a few epochs, lr is probably too low; if it explodes (NaN), too high.

### Batch size and epochs

Batch size 32 to 128 is the standard range. Larger batches → smoother gradient estimate but fewer updates per epoch and more memory. Smaller batches → noisier but more updates. `batch_size=64` is a fine starting point.

Number of epochs: AG News with this pipeline tends to converge in 5-20 epochs. Watch the loss — when it stops dropping for a couple epochs, stop. Or set `epochs=10` and call it a day if the loss curve looks healthy.

### The five-line training loop

The skeleton in Cell 17 already has the loop. Read it once and burn the order into your retina, because if you ever swap two lines the model breaks silently:

```python
optimizer.zero_grad()                # 1. clear stale grads from last step
outputs = model(X_batch)             # 2. forward pass
loss = criterion(outputs, y_batch)   # 3. compute loss
loss.backward()                      # 4. backprop — fill .grad on every parameter
optimizer.step()                     # 5. apply the update
```

If `zero_grad` is missing, gradients accumulate from previous batches and training diverges. If `backward` comes after `step`, you're applying the update before computing what to update. **Order is law.**

---

## 11. Reading a confusion matrix

Accuracy alone is a single number. The **confusion matrix** is the next level of detail. Rows are true labels; columns are predicted labels. The diagonal is correct predictions. Everything off-diagonal is an error, and the error tells a story.

| True ↓ / Pred → | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|
| **World** | 1820 | 15 | **85** | 30 |
| **Sports** | 10 | 1885 | 5 | 0 |
| **Business** | **110** | 4 | 1750 | 36 |
| **Sci/Tech** | 40 | 2 | 50 | 1808 |

Read the off-diagonals. The big numbers in bold: **World ↔ Business**. Both 110 and 85. The model can't reliably tell them apart — and it shouldn't surprise you, because financial news is often international news. The "G7 summit" might be either. That's a real-world pattern your model has discovered, not a bug. The 50 Sci/Tech → Business confusions are similar: tech earnings are both. The 36 Business → Sci/Tech are likely "Apple unveils new iPhone" headlines.

### Three statistics from one matrix

- **Precision** = TP / (TP + FP) = "of everything I called X, how much was X?"
- **Recall** = TP / (TP + FN) = "of everything that was X, how much did I find?"
- **F1** = harmonic mean of precision and recall = a single number that punishes asymmetry.

For AG News (balanced classes, 4 of them), accuracy and macro-F1 are usually close. If they diverge, you have one bad class dragging the average — investigate.

> **Side note — Quick sanity check**
> For a 4-class balanced problem, random chance is 25 %. Always-predict-the-majority is also 25 %. If your accuracy is below 60 % something is wrong with the pipeline. Above 85 % is good. State of the art (BERT-fine-tuned) is ~95 %.

---

## 12. Error analysis & interpretability

Task 1.4 asks for two things beyond the metrics. First: **look at three errors** and find the pattern. Second: **name a method** for figuring out which words in the input the model paid attention to.

### Reading the errors

Print headlines where `predicted ≠ true`. Look for these patterns:

- *Genuinely ambiguous* — "Microsoft posts Q3 earnings" could be Business or Sci/Tech.
- *Tokenization noise* — proper nouns split into nonsense pieces, losing signal.
- *Short headlines* — 4-5 tokens after BPE may not have enough information for the pooled vector to point anywhere meaningful.
- *Class-vocabulary overlap* — a Business headline that uses unusual sports metaphors ("market rebound", "comeback rally").

### How to find which words mattered

You don't have to implement this for Task 1.4 — you just have to **name a method and explain the idea**. Several reasonable choices:

| Method | Idea, in one sentence |
|--------|----------------------|
| **Token ablation** | Remove one token at a time and re-run; the drop in the predicted-class probability is that token's "importance". |
| **Gradient × input** | Compute $\partial \text{logit}_{\text{class}} / \partial v_i$ for each token vector; tokens with large gradient magnitude were influential. |
| **Integrated gradients** | Average gradient × input along a path from a "neutral" baseline (zero vector) to the actual input. More robust than raw gradients. |
| **LIME** | Train a tiny linear classifier locally around the input by perturbing words and observing predictions; report its weights as importances. |
| **Attention weights** | If you swapped pooling for an attention layer, the attention scores themselves are token importances. |

For a mean-pooled MLP, **token ablation** is the easiest and most honest: rebuild the sentence vector without one token at a time and watch how the output probabilities shift. Whichever token's removal changes the prediction most was the one the model leaned on.

> **A confusion matrix tells you what; an ablation tells you why.**

---

## 13. Char-level language modeling

Part 2 swaps the goal from "classify the document" to "predict the next character". Still cross-entropy. Still gradient descent. The only new piece is that the model has memory: a hidden state that travels from one character to the next.

The training task — formally — is autoregressive next-token prediction. Take the string `"<triceratops>"`. After seeing `<`, predict `t`. After `<t`, predict `r`. After `<tr`, predict `i`. And so on. The model learns the spelling and rhythm of the dinosaur language without ever being told the rules.

### The start & end tokens

Every name is wrapped in special tokens — `<` for start and `>` for end. After preprocessing, `"anna"` becomes `"<anna>"`. After concatenating the whole dataset, the long string looks like:

```
<allosaurus><ankylosaurus><anatotitan><archaeopteryx>...
```

At generation time, you feed `<` as the seed and stop sampling when the model emits `>`. The angle brackets are how the model says "I'm done".

### Why a sliding window

A 600-name concatenated string is too long to pass through an LSTM in one go. Instead, you slide a window of length `seq_len` (e.g. 30) over the string. Each window is one training example. The target for window at position `i` is the same window shifted right by one character.

Window of length 4 over `"<anna><raven>"`:

```
i=0:  x = <ann   y = anna
i=1:  x = anna   y = nna>
i=2:  x = nna>   y = na><
i=3:  x = a><r   y = ><ra
...
```

---

## 14. A PyTorch Dataset for sliding windows

A custom `torch.utils.data.Dataset` needs three methods: `__init__`, `__len__`, and `__getitem__`. For Task 2.1, the constructor does all the heavy lifting; the other two are one-liners.

### Step by step

1. Read the file into a list of names (one per line).
2. Lowercase if you want a smaller vocab (optional, but helpful).
3. Wrap each name in `<` and `>`.
4. Concatenate everything into one long string `self.text`.
5. Build the vocab: `set(self.text)` — this gives every unique char including the brackets.
6. Build `token_to_id` (char → int) and `id_to_token` (int → char).
7. Convert the whole text to a long list/tensor of integer IDs (`self.data`).

### `__len__` and `__getitem__`

How many windows of length `seq_len` fit into a string of length `L`? You need `seq_len + 1` consecutive characters per window (because `y` is shifted by 1). So:

$$|\text{Dataset}| = L - \text{seq\_len}$$

And `__getitem__(idx)` returns the slice `(self.data[idx : idx+seq_len], self.data[idx+1 : idx+seq_len+1])` as PyTorch `LongTensor`s.

> **Side note — Be careful with stride**
> Default sliding window with stride=1 generates overlapping windows. That's fine. Some implementations use stride=seq_len for non-overlapping windows. The assignment doesn't specify; stride=1 is a safe choice.

### Train/val split & DataLoader

After the Dataset is built, the assignment asks for a train/val split. The simplest path is `torch.utils.data.random_split` with an 80/20 ratio, then wrap each in a `DataLoader` with `shuffle=True` for train and `shuffle=False` for val. Use a fixed batch size — 64 matches the provided `train()` default.

**Pitfall worth flagging early:** the provided training loop uses `batch_size*n_steps` when reshaping targets — so your DataLoader's batch_size and the dataset's `seq_len` have to be exact, fixed numbers. If your last batch is smaller (because the dataset isn't divisible), the reshape will crash. Pass `drop_last=True` to the DataLoader.

---

## 15. One-hot encoding, slowly

The CharRNN doesn't have an embedding layer. Instead, each character ID is converted to a one-hot vector before being fed to the LSTM. If the vocab has 30 unique characters and the character is `'b'` with ID 2, the one-hot is a vector of 30 zeros with a single 1 at index 2.

```
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Why one-hot instead of an embedding? It works but is wasteful — every char is identical distance from every other in the input space, leaving all the structural learning to the LSTM. It's a deliberate teaching choice: this assignment wants you to feel the information bottleneck before reaching for embeddings in the next homework.

### Shapes are everything

The function takes an array of shape `(initial_array_shape)` and returns one of shape `(initial_array_shape, vocab_size)`. Concretely, your input from the DataLoader is shape `(batch_size, seq_len)`, and the output you need is `(batch_size, seq_len, vocab_size)`.

Cleanest implementations use NumPy's advanced indexing or `np.eye(vocab_size)[ids]` — the identity matrix indexed by IDs is a one-hot for each ID, and indexing broadcasts naturally over arbitrary shapes. Verify with a tiny toy input first; getting shapes wrong here will produce confusing LSTM errors later.

> **Side note — The PyTorch alternative**
> `torch.nn.functional.one_hot` does this in one line. The assignment asks for a custom NumPy implementation, presumably to make sure you understand what the operation is doing.

---

## 16. LSTMs — why a gate fixes a memory

A vanilla RNN updates its hidden state by smashing the old hidden state and the new input together with a single matrix multiply: $h_t = \tanh(W h_{t-1} + U x_t + b)$. This works for short sequences and falls apart for long ones, because the gradient of $h_T$ with respect to $h_1$ involves multiplying $W$ by itself $T$ times. If $\|W\| < 1$ the gradient vanishes exponentially. If $\|W\| > 1$ it explodes.

The **LSTM** fixes this with two ideas:

1. A separate **cell state** $c_t$ that carries memory along a "highway" with mostly additive updates (so gradients flow back without multiplying through tanh repeatedly).
2. **Gates** — small sigmoid-output networks that decide, at every step, what fraction of old memory to keep, what fraction of new input to write, and what fraction to read out.

### The four gates

| Gate | Job | Output |
|------|-----|--------|
| Forget $f_t$ | How much of old cell state to keep? | $\sigma(W_f [h_{t-1}, x_t])$ |
| Input $i_t$ | How much new candidate to write? | $\sigma(W_i [h_{t-1}, x_t])$ |
| Candidate $\tilde{c}_t$ | What new info to consider writing | $\tanh(W_c [h_{t-1}, x_t])$ |
| Output $o_t$ | How much of cell state to expose as hidden | $\sigma(W_o [h_{t-1}, x_t])$ |

The cell state and hidden state then update:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

Read this as: "Take the old memory, multiply elementwise by the forget gate (let some of it through), then add the new candidate scaled by the input gate. The visible hidden state is the cell state filtered by the output gate."

### What you actually pass to PyTorch

For your purposes, the gate machinery is hidden inside `nn.LSTM`. You pass:

- Input `x`: shape `(batch, seq_len, vocab_size)` when `batch_first=True`
- Initial hidden state tuple `(h_0, c_0)`: shape `(n_layers, batch, n_hidden)` each

And you get back:

- Output sequence: shape `(batch, seq_len, n_hidden)` — hidden state at every timestep
- Final hidden tuple `(h_T, c_T)`: shape same as input — state after the last step

### The reshape trick before the FC layer

The provided model has this line that throws students off:

```python
x = x.contiguous().view(x.size()[0]*x.size()[1], self.n_hidden)
```

It collapses the batch and time dimensions: `(batch, seq_len, n_hidden)` → `(batch*seq_len, n_hidden)`. The linear layer can then be applied to every timestep at once, producing `(batch*seq_len, vocab_size)` logits. Matching target shape is `(batch*seq_len,)` after flattening `y` with `.view(batch_size*n_steps)`.

### Detaching the hidden state

You'll see this in the training loop: `h = tuple([each.data for each in h])`. It strips the gradient history from the hidden state between batches. Without it, backpropagation would try to flow through the entire training history. With it, gradients only propagate within the current batch's `seq_len`. This is **truncated backpropagation through time** — universal in RNN training.

> **Side note — For Task 2.3**
> Run a single batch through the model step by step and print the shape after each line of `forward`. You should see: input `(64, 30, vocab_size)` → LSTM output `(64, 30, 256)` → reshape `(1920, 256)` → FC `(1920, vocab_size)` → softmax `(1920, vocab_size)`. If any of these don't match, your one-hot or batch dimensions are off.

---

## 17. Generation: greedy, random, top-k

Once trained, the model produces a probability distribution over the vocab at every step. The question is: how do we turn that distribution into a single chosen character?

### Greedy: pick the max

Simplest possible: `argmax(probabilities)`. The same input always produces the same output. Tends to produce repetitive and "safe" text — once the model is in a common pattern it never escapes.

### Random: sample by probability

`np.random.choice(vocab, p=probs)`. The provided `predict` function does this. The model can pick any character, weighted by its probability — characters with 0.001 probability still get picked occasionally. This produces variety but also occasional gibberish, because rare characters are usually rare for a reason.

### Top-k: random, but only among the top k

The compromise. Sort the probabilities, keep only the top `k`, renormalize so they sum to 1, and sample from that smaller distribution. With `k=5`, the model must choose from the 5 most likely characters at every step.

The effect: variety like random sampling, but no chance of catastrophic gibberish from the long tail. Most modern generation uses some form of truncated sampling (top-k or nucleus / top-p).

### Visual: top-k truncation

Suppose the model produces probabilities `[0.32, 0.22, 0.16, 0.10, 0.08, 0.06, 0.04, 0.02]` over 8 characters. With `k=3`:

```
char     :  s    a    u    r    o    n    i    x
prob     : 0.32 0.22 0.16 0.10 0.08 0.06 0.04 0.02
        :  ███  ███  ███   .    .    .    .    .     ← keep top-3
renorm  : 0.46 0.32 0.23                            ← then sample from these
```

---

## 18. Temperature — the variance dial

Temperature is a single number $T$ that you divide the logits by before applying softmax. It reshapes the probability distribution without changing the relative ordering.

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Three regimes to memorize:

- **$T < 1$** (e.g. 0.5): logits are amplified; the distribution becomes sharper; high-probability characters get even higher probability. Output becomes more "confident", more repetitive.
- **$T = 1$**: identity. The distribution is unchanged.
- **$T > 1$** (e.g. 1.5, 2.0): logits are squashed; the distribution becomes flatter; rare characters get a fairer shot. Output becomes more diverse, more surprising, and at high T, more chaotic.

### The math, with numbers

Suppose the model produces logits $z = [3.0, 2.0, 1.0]$ for three characters.

| T | Scaled logits | Softmax |
|---|--------------|---------|
| 1.0 | `[3.0, 2.0, 1.0]` | `[0.665, 0.244, 0.090]` |
| 0.5 | `[6.0, 4.0, 2.0]` | `[0.867, 0.117, 0.016]` |
| 2.0 | `[1.5, 1.0, 0.5]` | `[0.506, 0.307, 0.187]` |

At $T = 0.5$ the lead character is overwhelming. At $T = 2.0$ the underdogs become realistic options.

### Top-k and temperature together

Top-k and temperature compose. Apply temperature first to reshape the distribution, then top-k to truncate the long tail. Common settings for creative text: `T = 0.8, k = 40`. For factual / safer outputs: `T = 0.5, k = 5`. For your dinosaur names, anything in `T ∈ [0.6, 1.2], k ∈ [3, 10]` is worth comparing.

### What to write up for Task 2.6

Generate ~10 names at each setting, by eye:

- *Random*: noisy baseline. Some pronounceable, some odd.
- *Top-k = 3*: very repetitive, mostly looks like the most common dinosaur suffixes.
- *Top-k = 5, T = 1.0*: variety with structure. Sweet spot for plausibility.
- *Top-k = 5, T = 0.5*: extremely consistent, possibly memorized.
- *Top-k = 5, T = 1.8*: diverse and weird; some implausible.

Your insight should articulate the **diversity ↔ plausibility tradeoff**: lower T and lower k produce more "real-looking" but more repetitive names; higher T and larger k produce more original but less plausible names. **The "creativity" of the model is largely decided at sampling time, not training time.**

> **Temperature is the model's volume knob; top-k is its quality control.**

---

## 19. The ten tasks, distilled

### PART 1 · Task 1 — Tokenization (BPE & WordPiece) — 10 pts

Code already provided. Your job: review what's happening, inspect the example outputs, answer the conceptual question.

**Concepts you need:**
- What subword tokenization solves vs word-level (§2)
- BPE merge rule: most frequent pair (§3)
- WordPiece merge rule: highest likelihood ratio (§4)
- The role of `##` as continuation marker

**What to write:**
- One paragraph: BPE = frequency-greedy, WordPiece = likelihood-based
- When to prefer each (autoregressive vs bidirectional ecosystems)
- Why both beat word-level: no `[UNK]`, smaller fixed vocab, morphology shared

---

### PART 1 · Task 2 — Train Word2Vec embeddings — 10 pts

Train 6 models (CBOW × {50, 100, 200} and Skip-gram × {50, 100, 200}). Inspect 3 example tokens and their nearest neighbors. Write the analysis.

**Concepts you need:**
- The distributional hypothesis (§5)
- CBOW vs Skip-gram — direction of prediction (§6)
- Cosine similarity as the natural distance metric
- What `min_count` does to the learned vocab (§7.3)

**Thinking required:**
- Build `tokenized_sentences` by encoding each `train_df.text`
- Pick one model as "best" — usually Skip-gram + 100d for this dataset size
- Why does the W2V vocab differ from the tokenizer's 10k? Answer: `min_count=2`
- Justify the dimension choice with a one-line corpus-size rule

---

### PART 1 · Task 3 — MLP architecture & training — 10 pts

Tokenize both splits → average their embeddings → feed pooled vectors to MLP → cross-entropy → Adam → 5-20 epochs.

**Concepts you need:**
- Pooling strategies and what each loses (§8)
- Why ReLU and why deep ≠ better here (§9)
- The four-line training loop in canonical order (§10)
- Why MLP works on pooled embeddings but cannot model order

**Thinking required:**
- Choose pool: mean is the safe default; justify if you pick something else
- 1 hidden layer of size ~256, ReLU, output 4 — start there
- Adam, lr=1e-3, batch=64, epochs=10 — defensible defaults
- Articulate the limitation: bag-of-embeddings = no order; RNN/Transformer fixes that

---

### PART 1 · Task 4 — Evaluation & error analysis — 20 pts

Highest-value task in Part 1 — don't shortchange the analysis.

**Concepts you need:**
- How to read a confusion matrix off-diagonal (§11)
- Precision / Recall / F1 from one matrix
- Common error patterns: ambiguity, short headlines, vocab overlap (§12)
- One interpretability method — token ablation is the simplest

**Thinking required:**
- Use `sklearn.metrics.confusion_matrix` and a heatmap
- Print 3 misclassifications with full text + true + predicted
- Identify a pattern (Business ↔ World is the classic AG News confusion)
- Name token ablation, gradient × input, integrated gradients, or LIME — explain the idea in 2 lines

---

### PART 2 · Task 1 — `DinosDataset` (sliding window) — 10 pts

Custom Dataset that loads names, wraps each in `<`/`>`, builds a vocab, and yields fixed-length sliding windows.

**Concepts you need:**
- Why `<` / `>` markers (§13)
- Sliding window and why `y` is shifted by one (§13)
- The 7-step constructor (§14)
- `__len__ = L − seq_len`

**Thinking required:**
- Detect "is data a path" with `isinstance(data, str)` + `os.path.exists`
- Use `set(self.text)` for the vocab
- Return `LongTensor`s, not numpy arrays, from `__getitem__`
- Pass `drop_last=True` to the DataLoader to avoid the ragged-batch crash

---

### PART 2 · Task 2 — One-hot encoding — 5 pts

Five-line function. Easy points.

**Concepts you need:**
- One-hot is identity matrix indexed by IDs (§15)
- Output shape: `(input_shape, vocab_size)`

**Thinking required:**
- `np.eye(vocab_size)[array]` is a one-line solution
- Test on a small array first; verify shape
- Return numpy (not tensor) — the train loop wraps it

---

### PART 2 · Task 3 — Forward-pass walkthrough — 5 pts

Take one batch from the loader. Run it manually through the LSTM, dropout, reshape, FC layer, printing shape after each.

**Concepts you need:**
- LSTM input shape: `(B, L, V)` with `batch_first=True` (§16)
- The `(B, L, H)` → `(B*L, H)` reshape trick
- Hidden state is initialized as zeros via `init_hidden`

**Thinking required:**
- Get one batch with `next(iter(train_loader))`
- One-hot encode `batch[0]`
- Call `model.init_hidden(batch_size, 'cpu')` for the initial state
- Print shapes; apply `F.softmax(out, dim=1)` for probabilities

---

### PART 2 · Task 4 — Run training — 10 pts

Initialize a CharRNN, call the provided `train()` function with sensible hyperparameters, let it run.

**Concepts you need:**
- Truncated BPTT and why hidden state is detached (§16)
- Gradient clipping (already in `train()`) prevents RNN explosions
- What a healthy loss curve looks like (smooth decline, no NaNs)

**Thinking required:**
- Pass `vocab_size = len(train_data.vocab)`
- Reasonable defaults: `n_hidden=256, n_layers=2, lr=0.001, epochs=20`
- Make sure your DataLoader batch_size matches what you pass to `train()`
- If loss goes NaN: check one-hot shape, check your seq_len

---

### PART 2 · Task 5 — Generate names — 5 pts

Call `sample(model, size=...)` with several different starting prefixes.

**Concepts you need:**
- The default `predict` uses random sampling from softmax
- The model stops naturally on `>` or after `size` chars

**Thinking required:**
- Try prefixes like `'<a'`, `'<d'`, `'<ty'`, `'<m'`
- Generate ~10 names; pick the most charming ones for the report
- Note which prefixes produced more plausible-looking results

---

### PART 2 · Task 6 — Top-k & temperature — 15 pts

Modify `predict` and `sample` to add `top_k` and `temperature` parameters. Compare generations across settings. Write the diversity-vs-plausibility analysis.

**Concepts you need:**
- Top-k truncation: zero out everything outside the top k, renormalize (§17)
- Temperature: divide logits by T before softmax (§18)
- Three regimes: T < 1 (sharper), T = 1 (identity), T > 1 (flatter)
- Top-k and temperature compose

**Thinking required:**
- Pass `top_k` and `temperature` through both functions
- For top-k: sort, take indices, build a new probability vector with only those
- For temperature: divide raw logits before softmax (apply in `predict`, not on the already-softmaxed output)
- Run a 3×3 grid: `k ∈ {3, 5, ∞}` × `T ∈ {0.5, 1.0, 1.5}`
- Insight: low T + low k → repetitive but plausible; high T + large k → diverse but weird

---

*Set in Fraunces & Newsreader. Math by KaTeX. Written for someone who learns by understanding, not by copying. Hometask 02 · Field Guide · Tel Aviv 2026.*
