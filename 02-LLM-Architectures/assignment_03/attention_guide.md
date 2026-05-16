# A Field Guide to Attention

*For LLM Architectures Hometask 3, Part 2 (25 points)*

On how a transformer decides which tokens belong together — three matrices, one division by √d, a softmax, and the entire reason any of this works.

**Tools:** PyTorch · matplotlib · seaborn
**Read time:** ~25 minutes

---

## Contents

1. [The Problem Attention Solves](#1-the-problem-attention-solves)
2. [Q, K, V — Three Roles for the Same Token](#2-q-k-v--three-roles-for-the-same-token)
3. [The Scaled Dot-Product Formula](#3-the-scaled-dot-product-formula)
4. [Why Divide by √d_k](#4-why-divide-by-d_k)
5. [Reading a Heatmap](#5-reading-a-heatmap)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Masking](#7-masking)
8. [Task Cards](#8-task-cards)

---

## 1. The Problem Attention Solves

> *A sentence is a graph of relationships pretending to be a sequence. Attention is how a model rebuilds the graph.*

Consider the sentence "*the animal didn't cross the street because it was too tired*." The word **it** refers to **the animal**, not the street. You and I know this from semantics. A model needs to *discover* this — to look at "it" and decide that the most useful context for predicting its meaning lives back at position 2.

Older architectures (RNNs, LSTMs) carried information forward in a hidden state. By the time they reached "it," the signal from "animal" had passed through eight nonlinear gates and had been mostly forgotten. Attention takes a different approach: at every position, look at *every* other position directly, and weight them by relevance.

> **Attention is a soft, learned lookup table.**

That sentence is the entire intuition. Attention is a weighted average — but the weights are not fixed; they are computed on the fly from the content of the tokens themselves.

---

## 2. Q, K, V — Three Roles for the Same Token

> *Each token plays three parts in attention. Query: "what am I looking for?" Key: "what do I advertise?" Value: "what do I give to whoever picks me?"*

Every token in your sequence is represented by a vector (an embedding). Attention projects each embedding into three new vectors via three learned matrices:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $X$ is the matrix of input embeddings and $W^Q, W^K, W^V$ are learnable projection matrices.

| Vector | Role | Library analogy |
|---|---|---|
| **Q** (Query) | "What is this token looking for in the rest of the sentence?" | The search term you type at the library counter. |
| **K** (Key) | "What does this token advertise about itself, so others can find it?" | The label on each book's spine. |
| **V** (Value) | "If someone picks me, what content do I hand over?" | The actual contents of the book. |

You match queries against keys (the dot product), then use the resulting weights to mix values together. Same token, three roles. The decoupling is what gives attention its expressive power.

*In* **self**-attention, X is both the source and the target — every token attends to every other token in the same sequence (including itself). In **cross**-attention, Q comes from one sequence and K, V come from another (e.g., decoder queries the encoder).

In Task 2.2 of the assignment, you initialize Q = K = V = embeddings (no projection matrices). That is still valid attention — it just means each token's query, key, and value are identical. The shapes work; the heatmap is meaningful. In Task 2.3 Experiment 2, you add the projection matrices and create multiple heads.

---

## 3. The Scaled Dot-Product Formula

> *One line of math, four operations. The most important equation in the transformer.*

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Reading it left to right:

1. **`Q K^T`** — Score every query against every key. The result is an `n × n` matrix of raw similarities.
2. **Divide by `√d_k`** — Rescale so the softmax behaves nicely (next section).
3. **`softmax`** — Convert scores into a probability distribution over keys, row by row. Now each row sums to 1.
4. **Multiply by `V`** — Take a weighted average of value vectors using those probabilities as weights.

### Pipeline shapes

```
   Q          Kᵀ            scores         (÷ √d_k)
 ┌────┐    ┌──────────┐    ┌──────────┐
 │n×dk│  · │ dk × n   │  = │  n × n   │  ── softmax ──┐
 └────┘    └──────────┘    └──────────┘                │
                                                    ┌──┴───────┐
                                                    │ weights  │
                                                    │  n × n   │
                                                    └────┬─────┘
                                                          ·  V (n × d_v)
                                                          =  output (n × d_v)
```

### A tiny worked example

Suppose seq_len = 2 and d_k = 2. The two tokens have embeddings:

| token | embedding |
|---|---|
| 1 (*cat*) | [1.0, 0.0] |
| 2 (*mat*) | [0.0, 1.0] |

Q = K = V = these embeddings. Compute Q · Kᵀ:

```
scores = [[1·1 + 0·0,  1·0 + 0·1],
          [0·1 + 1·0,  0·0 + 1·1]]
       = [[1, 0],
          [0, 1]]
```

Each token's query has dot product 1 with itself (parallel vectors) and 0 with the other (orthogonal). Scale by √2:

```
scaled = [[0.707, 0.000],
          [0.000, 0.707]]
```

Apply softmax row-wise — row 1 becomes [exp(0.707), exp(0)] normalized = [0.670, 0.330]:

```
weights = [[0.670, 0.330],
           [0.330, 0.670]]
```

Each token attends ~67% to itself, ~33% to the other. Multiply by V to get the output.

With random embeddings (as in Task 2.2), the heatmap will look roughly like this — a strong diagonal because each token's Q matches its own K best.

*Now imagine 6 tokens and d_k = 16. The structure is the same; you just have a 6 × 6 weight matrix to visualize.*

---

## 4. Why Divide by √d_k

> *A subtle but critical detail. Without it, training the transformer was impossible at scale.*

Suppose Q and K vectors have entries with mean 0 and variance 1. Their dot product is a sum of d_k products:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

The expected value is 0, but the **variance grows linearly with d_k**. With d_k = 64, dot products will land somewhere in roughly [−8, +8]. With d_k = 256, they reach ±16. Without scaling, the softmax sees inputs that are far too peaked.

### What softmax does to a peaked input

```
INPUT [8, 0, 0, 0]    →    SOFTMAX → [0.999, ~0, ~0, ~0]
  (no scaling)                        ← all mass on one
                                       ← gradient ≈ 0 → training dies

INPUT [1, 0, 0, 0]    →    SOFTMAX → [0.475, 0.175, 0.175, 0.175]
  (with √d_k = 8)                     ← balanced
                                       ← gradient flows → training works
```

Dividing by √d_k brings the variance back to 1 regardless of d_k. The softmax stays in the range where small changes in input produce meaningful changes in output — i.e., where the gradient is nonzero.

### The full picture

| Without scaling | With √d_k scaling |
|---|---|
| Variance of scores grows with d_k | Variance ≈ 1 regardless of d_k |
| Softmax saturates → one-hot weights | Softmax stays soft, smooth gradients |
| Gradient through softmax ≈ 0 | Gradient flows through all positions |
| Training stalls or never starts | Training proceeds normally |

> ⚠️ **The bug you can introduce here:** if you divide by `d_k` instead of `sqrt(d_k)`, you over-correct. Scores become too flat, attention weights become nearly uniform, and the model can't distinguish important from unimportant context. The *square root* is exactly what cancels the variance growth.

---

## 5. Reading a Heatmap

> *The attention weight matrix is a square heatmap with tokens on both axes. Rows are queries, columns are keys. Bright = "I'm paying attention here."*

For a sentence like "the cat sat on the mat" with 6 tokens, you get a 6 × 6 heatmap. Read it row by row: *row 3 ("sat") with bright column 2 ("cat")* means the word "sat" is attending heavily to "cat."

Schematic of a typical pattern (numbers are made up):

```
            the   cat   sat   on    the   mat
       ┌────────────────────────────────────────┐
  the  │ .95   .01   .01   .01   .01   .01  │
  cat  │ .03   .85   .07   .02   .01   .02  │
  sat  │ .05   .40   .45   .05   .02   .03  │   ← splits between "cat" and self
  on   │ .02   .05   .20   .60   .08   .05  │
  the  │ .10   .02   .02   .10   .70   .06  │
  mat  │ .02   .08   .03   .15   .22   .50  │
       └────────────────────────────────────────┘
       ← each row sums to 1 →
```

### Patterns to look for

- **Diagonal dominance**: each token attends to itself. Universal in untrained or near-identity attention.
- **Off-diagonal hotspots**: meaningful relationships (subject-verb, adjective-noun, pronoun-referent).
- **Vertical stripes**: one token everyone attends to. Often the first token, or a special `[CLS]`-like token.
- **Block structure**: tokens cluster into phrases.
- **Near-uniform**: attention has not learned anything; weights are essentially flat across the row. Common with random initialization (which is what you'll see in Task 2.2 with random embeddings — the heatmap will be noisy with a faint diagonal).

*With random embeddings (Task 2.2), don't expect interpretable patterns. The point is to verify your function runs and produces a valid probability matrix — rows summing to 1, all values in [0, 1].*

### Sanity checks

1. Each row sums to 1 (softmax property): `weights.sum(dim=-1)` should be all-ones.
2. All values are non-negative.
3. Shape is `(seq_len, seq_len)` for single-head, or `(heads, seq_len, seq_len)` for multi-head.
4. Changing a token's embedding changes the row and column corresponding to that token, but mostly leaves other rows alone.

---

## 6. Multi-Head Attention

> *One attention pattern is fine. Eight in parallel, each specialising, is much better.*

A single attention head computes one weighted average over the values. But sentences contain many *kinds* of relationships simultaneously: syntactic (subject-verb), semantic (synonyms), positional (adjacent words), referential (pronoun antecedents). A single head must compress all of these into one pattern.

Multi-head attention runs *h* attention computations in parallel, each with its own projection matrices:

$$\text{head}_i = \text{Attention}(X W^Q_i, X W^K_i, X W^V_i)$$

The outputs are concatenated and passed through a final projection. With h = 8 heads and d_model = 512, each head operates on a 64-dimensional slice: same compute as one big attention, but the representation is partitioned and each part is free to specialise.

```
         ┌──────────────────────┐
         │   HEAD 1: W^Q₁ W^K₁ W^V₁  │ → attn₁
   X  ─→ │   HEAD 2: W^Q₂ W^K₂ W^V₂  │ → attn₂  ──→ CONCAT ──→ W^O ──→ Z
         │   HEAD 3: W^Q₃ W^K₃ W^V₃  │ → attn₃
         │   ...                 │
         └──────────────────────┘
```

### What different heads tend to learn

In trained transformers, studies have found heads that attend to:

- The previous token (positional / sequential)
- The next token (look-ahead, in encoders)
- Words that match in part of speech
- Subject-verb pairs (syntactic dependency)
- Pronouns and their antecedents (coreference)
- Tokens that share a topic or domain

For Task 2.3 Experiment 2: even with random projection matrices, your two heads will produce *different* heatmaps. The differences are random — not meaningful — but the structure is what matters. With training, those differences would become specialised.

> **Why multiple heads, not just one bigger head?**
> You could increase d_k for one head and get the same compute. But with one head, you compute exactly one weighted average. With h heads on d_k/h dimensions each, you compute h *different* averages and concatenate. The expressive power of "look at different relationships in parallel" requires the separation. Empirically, h = 8 or h = 16 outperforms a single big head at equal compute.

---

## 7. Masking

> *Sometimes you want a token to* not *see other tokens. The mask is how.*

The mask argument in `scaled_dot_product_attention` is for two cases:

1. **Causal masking**: in a decoder, token *t* may attend only to tokens at positions ≤ *t*. We don't let the model peek at the future during training.
2. **Padding masking**: when batches contain sequences of different lengths, shorter sequences are padded. The pad tokens should be ignored.

### How masking works

You don't multiply by the mask. You *add* a large negative number to the scores at the masked positions, *before* softmax:

$$\text{scores}_{ij} = \begin{cases} \text{scores}_{ij} & \text{if not masked} \\ -\infty & \text{if masked} \end{cases}$$

After softmax, `exp(−∞) = 0`, so those positions contribute zero weight. In code: `scores.masked_fill(mask == 0, float('-inf'))`.

> 💡 **Why -∞, not zero?**
> If you zeroed out the scores instead of pushing them to -∞, the softmax would still assign them `exp(0) = 1` of unnormalized mass — far from zero. You have to send the score below the softmax's effective range, and -∞ is the limit.

For Task 2.1, your skeleton has a mask argument with `pass`. Implement it — even if the experiments don't use it, a correct implementation should handle the mask path.

---

## 8. Task Cards

Each task in Part 2, with the concepts and the thinking. No code — that's yours.

### Task 2.1 — Implement scaled dot-product attention (5 pts)

**Concepts needed:**
- Matrix multiplication in PyTorch: `torch.matmul` or `@`
- Transpose of the last two dims of K: `K.transpose(-2, -1)`
- `math.sqrt` for the scaling factor
- `F.softmax(..., dim=-1)` — softmax over the last axis (keys)
- `scores.masked_fill(mask == 0, float('-inf'))` for the mask

**Thinking:**
- What shape is `Q @ K.transpose(-2, -1)`? Confirm it's `(batch, heads, seq_len_q, seq_len_k)`.
- Do you scale before or after the mask is applied? Convention: scale first, then mask, then softmax.
- What dimension does softmax run over? It must be the last — the keys dimension.
- Verify: `output.shape == (batch, heads, seq_len_q, d_v)`.

### Task 2.2 — Visualize attention as a heatmap (5 pts)

**Concepts needed:**
- `sentence.split()` for tokenization
- `torch.randn(seq_len, d_model)` for random embeddings
- Your function from Task 2.1 — but note it expects 4D input. You need to `.unsqueeze(0).unsqueeze(0)` to add batch and head dims, then squeeze them back for plotting.
- `seaborn.heatmap` with `xticklabels=tokens, yticklabels=tokens`
- `plt.xticks(rotation=45)` if labels overlap

**Thinking:**
- The attention weights from Task 2.1 have shape `(1, 1, seq_len, seq_len)`. To plot, squeeze them: `weights.squeeze().detach().numpy()`.
- With random embeddings, what do you expect to see? A faint diagonal (each token attends most to itself), noise elsewhere. Not interpretable language patterns — that requires training.
- Choose a colormap with high contrast (`"viridis"`, `"magma"`) so weak signal is still visible.

### Task 2.3 — Experiments (15 pts)

**Concepts needed for Experiment 1 (change one word):**
- Re-run your Task 2.2 pipeline with a different sentence
- Plot two heatmaps side by side (`plt.subplots(1, 2)`)
- Set the random seed (`torch.manual_seed(...)`) so the comparison is fair — otherwise different random embeddings will dominate the difference

**Concepts needed for Experiment 2 (multiple heads):**
- Learned (here: randomly initialized) projection matrices: `W_Q = torch.randn(d_model, d_k)`, same for W_K, W_V
- Q = X @ W_Q, K = X @ W_K, V = X @ W_V — these now differ from X
- Two heads = two sets of independent W matrices, each producing a separate heatmap

**Thinking — for the three questions:**

1. **Did changing one word affect attention?** If you change "cat" to "dog" with the same random seed, the embeddings for that position change — so the row and column for "cat"/"dog" change. Other tokens' embeddings are unchanged, but their attention *weights* change because the keys they see are different. Discuss this.

2. **Do different heads focus on different tokens?** Yes — with different random W matrices, the projected Q, K, V are different, so the patterns differ. Even with random matrices. With learned matrices, the differences become meaningful (one head attends to subjects, another to local context, etc.).

3. **Why is multi-head useful in transformers?** Capacity to capture different relationship types in parallel. Each head specializes. One pattern can't represent syntax, semantics, and position simultaneously. The decomposition into multiple heads gives the model many partial views of the sequence at the cost no more than one large head would have.

---

*Guide 03B · Attention · LLM Architectures Hometask 3 · Part 2*
