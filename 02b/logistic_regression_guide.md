# A Field Guide to Logistic Regression

> A study companion for **LLM Architectures — Hometask 01, Part 1**.
> Written so that after reading it you can write the code yourself, and understand what every line is asking of you.

| | |
|---|---|
| **For** | LLM Architectures, Hometask 01 |
| **Covers** | Parts 1 & 2 — all 100 pts |
| **Reading time** | ~35 minutes, slow |
| **Promise** | Concepts only — no solutions |

---

## Contents

1. [What logistic regression actually is](#1--what-logistic-regression-actually-is)
2. [The sigmoid — an honest probability](#2--the-sigmoid--an-honest-probability)
3. [Binary cross-entropy & the meaning of surprise](#3--binary-cross-entropy--the-meaning-of-surprise)
4. [Numerical stability — the BCE side](#4--numerical-stability--the-bce-side)
5. [Numerical stability — the softmax side](#5--numerical-stability--the-softmax-side)
6. [Building the model in PyTorch](#6--building-the-model-in-pytorch)
7. [Why weight initialization matters (or doesn't)](#7--why-weight-initialization-matters-or-doesnt)
8. [Mini-batch SGD, told slowly](#8--mini-batch-sgd-told-slowly)
9. [Learning rate & batch size — two dials](#9--learning-rate--batch-size--two-dials)
10. [Choosing a metric you can defend](#10--choosing-a-metric-you-can-defend)
11. [Why we regularize at all](#11--why-we-regularize-at-all)
12. [L1 vs L2 — diamonds & circles](#12--l1-vs-l2--diamonds--circles)
13. [What to expect from your λ sweep](#13--what-to-expect-from-your-λ-sweep)
14. [The five tasks, distilled](#14--the-five-tasks-distilled)
15. [What an optimizer actually does](#15--what-an-optimizer-actually-does)
16. [The four optimizers, in order](#16--the-four-optimizers-in-order)
17. [Two landscapes — bowl & camel](#17--two-landscapes--bowl--camel)
18. [Reading your optimizer results](#18--reading-your-optimizer-results)
19. [The sixth task, distilled](#19--the-sixth-task-distilled)

---

## § 1 — What logistic regression actually is

Forget the name for a moment. Logistic regression is a tool for answering one question: **"given these features, what's the probability that this example belongs to class 1?"** It is misleadingly named — the word "regression" only refers to how it is fitted, not what it does. It is a classifier through and through.

Mechanically, it is two tiny pieces glued together. First, a *linear score*: take the input vector $x$, multiply it elementwise by a weight vector $w$, sum, and add a bias $b$. Call the result $z$, the **logit**:

$$z = w^\top x + b$$

Then, a *squashing function* — the sigmoid — that takes $z$ (any real number from $-\infty$ to $+\infty$) and pinches it down into a probability between $0$ and $1$:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

That's the entire model. To turn the probability into a hard prediction you pick a threshold (usually $0.5$). Anything above is class 1; anything below is class 0.

> **Sidenote — Why "linear"?**
> Because the boundary between predicted class 0 and predicted class 1 is the set of points where $\sigma(z) = 0.5$, which is where $z = 0$, which is the hyperplane $w^\top x + b = 0$. A flat slice through feature-space.

For your hometask the input $x$ is a 10,000-dimensional bag-of-words vector — a count of how often each vocabulary word appears in a movie review. The output is one number: the probability the review is positive. So your $w$ is a vector of 10,000 weights, one per vocabulary word, each saying "how much does seeing this word make me think positive vs. negative?" **A learned $w$ is a tiny lexicon of sentiment.**

> ### *Logistic regression is a lexicon you don't have to write by hand.*

---

## § 2 — The sigmoid — an honest probability

The sigmoid is the bridge from "any real number" to "valid probability". It looks like an elongated S laid on its side: flat at the bottom, steep through the middle, flat at the top. Three properties matter:

1. $\sigma(0) = 0.5$. Equal evidence both ways means equal probability.
2. It is symmetric around the origin: $\sigma(-z) = 1 - \sigma(z)$.
3. It **saturates** in the tails: $\sigma(10) \approx 0.99995$, $\sigma(-10) \approx 0.00005$. This will haunt us in §4.

### Reference table — sigmoid + BCE

| $z$ | $\sigma(z)$ | $-\log\sigma(z)$  (loss if $y=1$) | $-\log(1-\sigma(z))$  (loss if $y=0$) | saturated? |
|----:|------------:|------------------------------------:|-----------------------------------------:|:----------:|
| −6  | 0.0025      | 6.00                                | 0.0025                                   | yes |
| −3  | 0.0474      | 3.05                                | 0.0486                                   | no  |
| −1  | 0.2689      | 1.31                                | 0.3133                                   | no  |
|  0  | 0.5000      | 0.693                               | 0.693                                    | no  |
| +1  | 0.7311      | 0.3133                              | 1.31                                     | no  |
| +3  | 0.9526      | 0.0486                              | 3.05                                     | no  |
| +6  | 0.9975      | 0.0025                              | 6.00                                     | yes |

Notice how, once $z$ is past about $|4|$, the sigmoid is already crushing the probability into the last few digits of a float. By $|z| = 15$ it's effectively $0$ or $1$ exactly. The information content of "how confident am I?" has disappeared into underflow.

---

## § 3 — Binary cross-entropy & the meaning of surprise

We need a way to score "how wrong was the model on this example?" — a loss. For regression you reach for squared error. For classification with probabilities, the right tool is the **cross-entropy**, and there is a beautiful reason why.

Think of $-\log p$ as the *surprise* of seeing an event with probability $p$. If you predicted something with probability $0.99$ and it happens, $-\log(0.99) \approx 0.01$ — almost no surprise. If you predicted it with probability $0.01$ and it happens anyway, $-\log(0.01) \approx 4.6$ — large surprise. Surprise is a punishment proportional to how badly you misjudged the world.

The binary cross-entropy is just "the surprise of the true label, averaged over the batch":

$$L = -\frac{1}{N}\sum_{i=1}^N \Big[\, y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \,\Big]$$

Read the bracketed term carefully. Only one of the two pieces is "live" for any given example, because $y_i$ is either $0$ or $1$:

- If $y_i = 1$, the second term vanishes and you're left with $-\log \hat{y}_i$. Want $\hat{y}$ near 1.
- If $y_i = 0$, the first term vanishes and you're left with $-\log(1 - \hat{y}_i)$. Want $\hat{y}$ near 0.

> **Sidenote — Why not squared error?**
> With sigmoid + MSE the loss surface is non-convex and the gradient is tiny exactly where the model is most wrong (deep in saturation). BCE was designed to *cancel out* the sigmoid derivative when you take gradients, leaving a clean linear pull toward the right answer. It is the loss the sigmoid was made for.

Look back at the table in §2. The two loss columns are exactly $-\log\hat{y}$ and $-\log(1-\hat{y})$. At the extremes, the loss grows fast — and when the sigmoid saturates fully, the loss becomes infinite. That's the enemy in §4.

---

## § 4 — Numerical stability — the BCE side

Floating-point numbers are not real numbers. They have limits. A 32-bit float can only represent values down to about $1.4 \times 10^{-45}$ before it underflows to zero. When the sigmoid output gets close enough to $0$ or $1$, it *becomes* exactly $0$ or exactly $1$. And then:

$$\log(0) = -\infty \qquad \log(1 - 1) = \log(0) = -\infty$$

$-\infty$ multiplied by anything is still $-\infty$, and $-\infty$ summed into your loss gives you a NaN, and a NaN flowing through backpropagation **poisons your weights forever**. One bad batch and the entire run is ruined.

The cheap, robust fix: **clamp the prediction** away from the danger zone before you take the log. Pick a tiny $\varepsilon$ — the standard choice is $10^{-15}$ — and squeeze every $\hat{y}$ into the safe range $[\varepsilon,\ 1 - \varepsilon]$:

```text
y_pred = sigmoid(logits)
y_pred = clamp(y_pred, eps, 1 - eps)
loss   = -mean( y * log(y_pred) + (1 - y) * log(1 - y_pred) )
```

With clamping, the worst the model can be punished is roughly $-\log(10^{-15}) \approx 34.5$. Painful but finite. Training survives.

> **Bonus knowledge.** Production frameworks (PyTorch's `BCEWithLogitsLoss`) skip the clamp entirely by working directly on the logits, never materialising the saturated probabilities. The formula is rearranged into `max(z, 0) − z·y + log1p(exp(−|z|))`, which is stable for any $z$. For the hometask you implement the clamp version, but it's worth knowing why production code looks weird.

---

## § 5 — Numerical stability — the softmax side

Softmax is the multi-class generalisation of the sigmoid. Given a vector of logits $z = [z_1, z_2, \ldots, z_K]$, it produces a probability distribution over $K$ classes:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

Two things can go wrong with the naive formula. If any $z_i$ is large (say $1000$), $e^{z_i}$ **overflows** to infinity. If every $z_i$ is very negative, every $e^{z_i}$ underflows to zero, the denominator becomes zero, and you divide by zero. Either way: NaN.

The fix is the most quietly elegant trick in deep learning. Subtract the maximum logit from every logit before exponentiating:

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$

### i. Why this gives the same answer

Let $c = \max(z)$. We want to show that subtracting $c$ leaves the ratio unchanged. Multiply numerator and denominator by $e^{-c}$:

$$\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_j e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}$$

The constant factor cancels. **The transformation is exact algebra, not an approximation.**

### ii. Why this is suddenly stable

After subtracting the max, every $(z_i - c)$ is at most $0$. So every $e^{z_i - c}$ is in $(0, 1]$. The biggest exponent is $0$, giving $e^0 = 1$ — **no overflow is possible**. And because at least one term in the denominator is exactly $1$, the denominator is always at least $1$, so we **never divide by zero** either.

### Worked example

```text
NAIVE
  z   = [1000, 1001, 1002]
  e^z = [Infinity, Infinity, Infinity]
  → NaN / NaN = 💀

STABLE  (subtract max = 1002)
  z − c     = [−2, −1, 0]
  e^(z − c) = [0.135, 0.368, 1.000]
  softmax   = [0.090, 0.245, 0.665]
```

For the cross-entropy you take the log of softmax, which under the same trick becomes the **log-sum-exp** form:

$$\log \text{softmax}(z)_i = (z_i - c) - \log \sum_j e^{z_j - c}$$

The sum inside the log is now safe (always $\geq 1$), so the log never sees zero.

---

## § 6 — Building the model in PyTorch

`nn.Module` is PyTorch's base class for "anything that has learnable parameters and a forward pass." When you subclass it, two things matter most: `__init__` creates the learnable tensors, and `forward` defines how an input becomes an output. Everything else — gradient tracking, parameter registration, GPU movement — is handled for you, **as long as you wrap parameters in `nn.Parameter`**.

For a logistic regression you need exactly two learnable things:

- **w** — a weight tensor of shape `(n_features, 1)`. The trailing 1 makes the matrix multiply `x @ w` produce a column of logits, one per example.
- **b** — a bias tensor of shape `(1,)`. Broadcasts across the batch.

The `forward` pass is the model's mathematical definition translated to PyTorch ops. Three lines:

```text
def forward(self, x):
    logits = x @ self.w + self.b   # shape (batch, 1)
    probs  = torch.sigmoid(logits) # shape (batch, 1)
    return probs
```

The `predict` method exists only for evaluation: take the probabilities and threshold them at $0.5$ to get hard class labels. It is **never** called during training — the loss operates on probabilities, not on hard labels.

> **Sidenote — about `.clone().detach()`.**
> If a user passes a pre-built tensor as the initialization, you should clone-and-detach it before wrapping in `nn.Parameter`. Cloning ensures the parameter doesn't share memory with the caller's tensor (so future changes don't bleed in). Detaching breaks any computation graph the caller's tensor might be carrying. It's hygiene.

---

## § 7 — Why weight initialization matters (or doesn't)

In a deep network, initializing every weight to zero is broken — every neuron in a layer would compute the same thing, and the symmetry would never break under gradient descent. People panic about init for that reason. **For logistic regression that panic is misplaced.**

The BCE loss for a sigmoid linear model is **convex**. There is exactly one minimum. Gradient descent will find it from anywhere — including from all-zeros. So why would you ever pick random init? Two reasons:

1. **Speed.** A small random kick can put you in a slightly steeper part of the loss landscape on the first step, sometimes shaving a few epochs off training.
2. **Tie-breaking under regularization.** Once you add L1 (§12), the optimum may not be unique — a random init lets the optimizer commit to one solution rather than oscillating among several.

> ### *Random init for logistic regression is a luxury, not a survival tool.*

One warning that **does** apply: the random values must be **small**. If you initialize $w$ with values around 1, your initial logits could be in the tens or hundreds, the sigmoid saturates, the gradient (which is proportional to $\sigma(z)(1 - \sigma(z))$) effectively vanishes, and training crawls. The standard recipe is something like $\mathcal{N}(0, 0.01^2)$ — small enough to keep early logits in the linear part of the sigmoid.

---

## § 8 — Mini-batch SGD, told slowly

Gradient descent is one idea repeated forever: compute the gradient of the loss with respect to the parameters, take a small step in the opposite direction, repeat. The only question is "what data do you use to compute that gradient?" There are three answers:

- **Full-batch (vanilla GD).** Use the entire training set every step. The gradient is exact but expensive — one update per pass through the data.
- **Stochastic (pure SGD).** Use one example per step. Cheap and noisy. Each step is a random direction that's only roughly correct on average.
- **Mini-batch SGD.** The compromise everyone actually uses. Take a small random subset of $B$ examples, compute the gradient on that, step. Then again with a fresh subset.

Mini-batch wins almost everywhere. It is dramatically faster than full-batch (you take many steps per epoch), much less noisy than pure SGD (the average over $B$ examples is a decent estimate), and on a GPU it benefits from vectorisation — computing on $B$ examples is usually nearly as fast as computing on $1$.

The training loop pattern, in shape:

```text
for epoch in range(num_epochs):
    shuffle(training_indices)        # critical — see sidenote
    for batch in batches(training_indices, B):
        optimizer.zero_grad()        # don't accumulate yesterday's gradients
        y_hat = model(x_batch)
        loss  = bce(y_hat, y_batch)
        loss.backward()              # populate .grad
        optimizer.step()             # w ← w − lr * w.grad
    log(metric(model, x_val, y_val))
```

> **Sidenote — always shuffle.**
> If your data is sorted by class (all positives first, all negatives second), failing to shuffle means every batch is class-pure. The gradient on a class-pure batch points the wrong way as often as it points the right way, and training never converges. Shuffling once per epoch is the cheap insurance.

---

## § 9 — Learning rate & batch size — two dials

These two hyperparameters are the most consequential dials in any SGD run, and they interact. Understanding how is the entire point of **Task 1.4**.

### i. Learning rate

The learning rate $\eta$ scales every parameter update. Too small and training crawls — you spend most epochs barely moving. Too large and the loss oscillates or diverges outright (look for `nan` in your loss log; that's a divergence). The sweet spot is the **largest learning rate at which training is still monotonically decreasing on average**.

### ii. Batch size

The batch size $B$ controls how noisy your gradient estimate is. Smaller batches give noisier gradients, but more steps per epoch. Larger batches give smoother gradients but fewer steps per epoch. The variance of the gradient estimate scales like $1/B$.

### iii. The interaction

Here is the rule that ties them together: **larger batches tolerate larger learning rates**. Because a big-batch gradient is less noisy, you can take a bigger step in that direction without overshooting. The classic heuristic is the *linear scaling rule*: if you double $B$, you can roughly double $\eta$.

This means your hometask heatmap will **not** show one universal best corner. It will show a **diagonal ridge**: small batch with small lr is fine, large batch with large lr is fine, but the off-diagonal corners — small batch + large lr (divergence) and large batch + small lr (underfitting) — are where things go wrong.

### Expected heatmap shape (rough sketch)

```text
                    learning rate →
              0.01  0.03  0.1   0.3   1.0
batch  50  [  ok    ok   good  risk  NaN ]
 size  100 [ slow   ok   good  good risk ]
       200 [ slow  slow   ok   good good ]
```

> **Sidenote — what to look for.**
> Divergence shows up as a sudden drop to chance-level accuracy or as outright NaN. Underfitting shows up as accuracy noticeably worse than your best cell. The "good" cells should form a band running diagonally across the grid.

---

## § 10 — Choosing a metric you can defend

The hometask asks you to pick an evaluation metric and briefly justify it. Don't pick one because it sounds smart. Pick one because of what your data looks like. This section is longer than the others because the task explicitly grades you on the **justification**, not the choice — you need enough depth to write a defensible paragraph.

### i. The confusion matrix — foundation for everything

Every classification metric is built on a single 2×2 grid. For binary classification, each prediction falls into exactly one of four cells:

```text
                  Actual label
                positive    negative
Predicted  ┌──────────┬──────────┐
 positive  │    TP    │    FP    │
           ├──────────┼──────────┤
 negative  │    FN    │    TN    │
           └──────────┴──────────┘

TP = true  positive  (correctly called positive)
FP = false positive  (negative, wrongly called positive)
FN = false negative  (positive, wrongly called negative)
TN = true  negative  (correctly called negative)
```

Every metric is a different recipe on those four numbers. **Once you understand the grid, you understand every metric.**

### ii. A running example

Imagine a validation set of 100 reviews: 60 positive, 40 negative. Your model produces:

```text
TP = 50    FP = 10
FN = 10    TN = 30
```

Let's compute every metric from this one grid.

### iii. Accuracy — "what fraction did we get right?"

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{50 + 30}{100} = 0.80$$

Instantly interpretable — "80% correct." That is its only virtue. Its fatal flaw: **it collapses on imbalanced datasets**. If your data is 99% negative and 1% positive (fraud, medical screening, spam), a model that always predicts "negative" achieves 99% accuracy and has learned literally nothing. The trivial majority-class baseline dominates, and the metric cannot distinguish it from a real model.

> **Rule of thumb.** Accuracy is meaningful only when the classes are roughly balanced *and* you care about both kinds of error equally. Otherwise it's misleading.

### iv. Precision — "when we said positive, were we right?"

$$\text{precision} = \frac{TP}{TP + FP} = \frac{50}{50 + 10} = 0.833$$

Of every 100 times the model said "positive", about 83 of them actually were. Notice what precision *ignores*: false negatives. A model that predicts "positive" exactly once, and happens to be right, has precision = 100% — even if it missed a thousand other positives.

You care about precision when a **false positive is expensive**:

- **Spam filter** — flagging a real email as spam makes the user miss something important.
- **Investment recommendation** — recommending a bad stock loses money.
- **Content moderation** — removing legitimate posts causes user backlash.

### v. Recall — "of all actual positives, how many did we catch?"

$$\text{recall} = \frac{TP}{TP + FN} = \frac{50}{50 + 10} = 0.833$$

Of the 60 actual positive reviews, the model found 50. Recall ignores false positives: a model that predicts "positive" for everything has recall = 100% — and is useless because it flags every email as spam too.

You care about recall when a **false negative is expensive**:

- **Cancer screening** — missing a tumour is catastrophic; a false alarm just triggers another test.
- **Fraud detection** — missing real fraud costs money; a false alarm just annoys a legitimate customer.
- **Safety systems** — a missed warning can kill; a false alarm is merely noisy.

### vi. F1 — the harmonic mean of precision & recall

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = 2 \cdot \frac{0.833 \cdot 0.833}{0.833 + 0.833} = 0.833$$

The F1 score is the **harmonic mean** of precision and recall, not the plain arithmetic average. That choice is deliberate: the harmonic mean **punishes imbalance** between the two numbers.

**What is a harmonic mean?** The arithmetic mean is what you think of as "average":

$$\text{arith}(a, b) = \frac{a + b}{2}$$

The harmonic mean uses division instead of addition:

$$\text{harm}(a, b) = \frac{2}{\frac{1}{a} + \frac{1}{b}} = 2 \cdot \frac{a \cdot b}{a + b}$$

The second form on the right is what you get after simplifying — and it's exactly the F1 formula with `a = precision` and `b = recall`.

**Why it matters — compare two models:**

- **Model A** — precision 0.9, recall 0.9. Average = 0.9. F1 = 0.9. Both metrics honest.
- **Model B** — precision 1.0, recall 0.1. Average = 0.55 (sounds OK!). **F1 = 0.18** (correctly says "broken").

The arithmetic mean lets a great number (1.0) hide a terrible one (0.1). F1 refuses. The harmonic mean is always dragged toward the *smaller* of the two inputs, so **F1 tracks your bottleneck, not your best number**.

**Table — how F1 collapses when recall drops (precision held at 1.0):**

| precision | recall | arithmetic mean | **F1 (harmonic)** |
|----------:|-------:|----------------:|------------------:|
| 1.00 | 1.00 | 1.00 | **1.00** |
| 1.00 | 0.80 | 0.90 | **0.89** |
| 1.00 | 0.50 | 0.75 | **0.67** |
| 1.00 | 0.20 | 0.60 | **0.33** |
| 1.00 | 0.10 | 0.55 | **0.18** |
| 1.00 | 0.01 | 0.505 | **0.02** |

Notice how arithmetic mean stays near 0.5 because the 1.0 props it up, but F1 collapses toward the smaller value. That's the whole point.

Use F1 when you want a single number that stays honest across imbalanced datasets and you can't say in advance whether FP or FN is worse. It's a conservative default that's hard to game.

### vii. How to pick for your hometask dataset

Don't guess. Before you train anything, check the class balance of your actual data:

```python
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
print(f"positive: {n_pos} ({100*n_pos/len(y_train):.1f}%)")
print(f"negative: {n_neg} ({100*n_neg/len(y_train):.1f}%)")
```

SST-2 (the typical dataset for this hometask) is roughly **50/50 balanced**. That's the number you want to cite in your write-up.

**Decision tree for your justification:**

```text
Is the dataset roughly balanced (each class ≥ 40%)?
│
├── YES → accuracy is fine. F1 is also fine and more conservative.
│         Pick one, cite the balance, done.
│
└── NO  → accuracy is misleading. Use F1 / precision / recall.
          └── Do you care more about FP or FN?
              ├── FP is worse → precision
              ├── FN is worse → recall
              └── both / unsure → F1
```

### viii. Writing the defense paragraph

The hometask says "briefly explain why this metric is suitable". A minimum-viable answer has three beats: (1) the class balance you *observed*, (2) the metric you picked and why it fits, (3) what you'd do instead if the balance were different.

A template to adapt in your own words:

> *"SST-2 is roughly balanced (≈50% positive, ≈50% negative), so accuracy is meaningful: the 'always predict majority' baseline is not strong. I also compute F1 as a sanity check; it matches accuracy closely, confirming precision and recall are not diverging. If the dataset were imbalanced (say 90/10), I would switch to F1 to avoid rewarding trivial majority-class predictions."*

Three sentences. Shows you looked at the data. Shows you considered alternatives. Shows you can defend your choice. **That is what "briefly explain" means in a graded assignment.**

### ix. Computing metrics in PyTorch

You can roll them by hand from the confusion-matrix counts — which is a good exercise because it keeps the math visible:

```python
with torch.no_grad():
    preds = model.predict(x_val)

    tp = ((preds == 1) & (y_val == 1)).sum().item()
    fp = ((preds == 1) & (y_val == 0)).sum().item()
    fn = ((preds == 0) & (y_val == 1)).sum().item()
    tn = ((preds == 0) & (y_val == 0)).sum().item()

    accuracy  = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp + 1e-9)     # +eps avoids div-by-zero
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
```

Or let sklearn do it for you in one line:

```python
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(y_val.numpy(), preds.numpy())
f1  = f1_score(y_val.numpy(), preds.numpy())
```

Both give the same answer. The hand-rolled version is better for learning; the sklearn version is better once you know what you're doing.

### x. The key takeaway

The hometask is not testing whether you pick the "right" metric. It is testing whether you **thought about it** — whether you can look at your data, identify the relevant trade-off, and justify a choice. A confident three-sentence paragraph beats a vague one-liner every time, regardless of which metric you picked.

---

## § 11 — Why we regularize at all

Imagine your model has 10,000 features and 7,000 training examples. There are far more knobs than there are constraints. The model can — if you let it — find a setting of $w$ that fits the training data essentially perfectly, including all of the noise. Reviews with the word "Tuesday" happen to skew positive in your training set? The model learns to weight Tuesday. Test set, different reviews, no Tuesday signal: model misses.

That phenomenon is **overfitting**. The cure is to discourage the model from being too confident in any one feature. We do this by adding a **penalty** to the loss — a term that grows when the weights grow, so the optimizer has to balance "fit the data" against "keep weights modest."

$$L_{\text{total}}(w, b) = \underbrace{\text{BCE}(w, b)}_{\text{fit the data}} + \underbrace{\lambda \cdot \Omega(w)}_{\text{stay modest}}$$

The two common penalty functions $\Omega(w)$ are L2 (sum of squared weights) and L1 (sum of absolute weights). They look similar on paper. They behave very differently in practice.

$$\text{L2: } \;\Omega(w) = \sum_i w_i^2 \qquad\qquad \text{L1: } \;\Omega(w) = \sum_i |w_i|$$

> **Sidenote — don't penalize the bias.**
> The bias is just an intercept — it shifts the decision boundary toward or away from the origin. Penalizing it would push the boundary toward the origin for no good reason. Standard practice is to apply the penalty only to $w$.

---

## § 12 — L1 vs L2 — diamonds & circles

### i. The plain-language version

You have 10,000 words (features). Some are useful for predicting sentiment ("great", "terrible", "boring"). Most are noise ("Tuesday", "camera", "three"). You want the model to **ignore the noise words** by setting their weights to zero or near-zero. L1 and L2 are two different strategies for doing that, and they differ dramatically in outcome.

**L2 — "shrink everything evenly":** adds $\lambda \sum w_i^2$ to the loss. The pull toward zero is proportional to each weight's current size — big weights get pulled hard, tiny weights get pulled softly. Result: everything shrinks, but **nothing actually reaches zero**. You still have 10,000 non-zero weights, just smaller ones.

**L1 — "kill the unimportant ones":** adds $\lambda \sum |w_i|$ to the loss. The pull toward zero has **constant force**. Whether the weight is 0.5 or 0.001, L1 shoves it with the same strength. Small weights get dragged all the way to zero; large weights survive. **L1 does feature selection** — it decides which words matter and zeroes out the rest.

```text
Before regularization:
████████   big    ("great")
██████     medium ("film")
████       small  ("Tuesday")
██         tiny   ("camera")

After L2 — everything shrinks, nothing dies:
██████     smaller
████       smaller
███        smaller
█          smaller but still alive

After L1 — small ones die, big ones survive:
███████    slightly smaller
█████      slightly smaller
░          → effectively zero (killed!)
░          → effectively zero (killed!)
```

### i.b Concrete numbers

Suppose you have 5 weights before regularization:

```text
Before:  w = [0.80,  0.50,  0.05,  -0.03,  0.01]

L2 penalty for each:  w² = [0.64,  0.25,  0.0025,  0.0009,  0.0001]
  → L2 pulls 0.05 with force 0.05, pulls 0.01 with force 0.01
  → the small ones barely get pushed

L1 penalty for each: |w| = [0.80,  0.50,  0.05,   0.03,   0.01]
  → L1 pulls 0.05 with force 1.0 (sign), pulls 0.01 with force 1.0 (same!)
  → the small ones get pushed JUST AS HARD as the big ones

After many steps:
  L2: w ≈ [0.60,  0.38,  0.04,  -0.02,  0.008]   ← all smaller, none zero
  L1: w ≈ [0.65,  0.35,  0.00,   0.00,  0.000]   ← small ones died!
```

The key insight: L2's push is *proportional* to the weight (big weights feel it more). L1's push is *constant* (everyone feels the same force). Constant force + small weight = death.

### ii. The geometric view (the famous picture)

The most famous picture in all of regularization is the one that explains why L1 produces sparse solutions and L2 doesn't.

```text
        L1  (diamond)                       L2  (circle)

             w₂                                   w₂
              |                                    |
              ◆                                    ○
             ╱ ╲                                  ╱ ╲
        ───◆───◆─── w₁                       ───○───○─── w₁
             ╲ ╱                                  ╲ ╱
              ◆                                    ○
              |                                    |

    touch point = CORNER                touch point = generic point
    (one axis is zero)                  (no axis is zero)
        → sparsity                          → shrinkage only
```

Imagine you have a fixed budget for $\Omega(w)$. Under L1, the budget defines a diamond — a square rotated 45°, with corners on the coordinate axes. Under L2, it's a circle. The unconstrained loss minimum sits somewhere out beyond either region. To find the constrained best, you grow the loss contours from that center until they first **touch** the constraint region. Whichever point gets touched is your answer.

For the diamond, the touch point is overwhelmingly likely to be a **corner** — and corners lie on axes, meaning at least one $w_i = 0$. For the circle, the touch point is just some generic point on the boundary; nothing makes it prefer the axes. That's why L1 produces sparsity: **it pushes solutions to corners**. L2 just shrinks them.

### iii. The gradient view

The same story shows up in the gradients. The gradient of $|w_i|$ is $\text{sign}(w_i)$ — a constant-magnitude pull toward zero, no matter how small $w_i$ already is. The gradient of $w_i^2$ is $2 w_i$ — which *vanishes* as $w_i$ shrinks. So L1 keeps pressing tiny weights all the way to zero; L2 gives up before it gets there.

### iv. Why your SGD won't make weights exactly zero

Despite all this, when you train with mini-batch SGD on an L1-regularized loss, you'll find very few weights at *exactly* zero. Many will be tiny — $10^{-6}$, $10^{-8}$ — but not literally zero. This is because at every step, the noisy mini-batch gradient nudges weights slightly off zero just as the L1 penalty was about to clip them. To get true zeros you need **proximal methods** (e.g. ISTA, FISTA) that explicitly soft-threshold weights every step. The hometask asks you to count weights below $10^{-7}$ as a workaround — it captures the same pattern.

### Mini-demo: soft-thresholding by hand

Here's what the L1 penalty "wants" to do to a set of weights as $\lambda$ grows. Each row applies the soft-threshold operator $w \mapsto \text{sign}(w)\cdot\max(|w|-\lambda,\ 0)$ to a fixed set of eight hypothetical weights.

| λ     | w₁    | w₂    | w₃    | w₄    | w₅    | w₆    | w₇    | w₈    | non-zero |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------:|
| 0.00  | +0.82 | −0.45 | +0.07 | +0.61 | −0.18 | +0.04 | +0.95 | −0.72 | 8/8 |
| 0.05  | +0.77 | −0.40 | +0.02 | +0.56 | −0.13 | 0.00  | +0.90 | −0.67 | 7/8 |
| 0.15  | +0.67 | −0.30 | 0.00  | +0.46 | −0.03 | 0.00  | +0.80 | −0.57 | 6/8 |
| 0.30  | +0.52 | −0.15 | 0.00  | +0.31 | 0.00  | 0.00  | +0.65 | −0.42 | 5/8 |
| 0.50  | +0.32 | 0.00  | 0.00  | +0.11 | 0.00  | 0.00  | +0.45 | −0.22 | 4/8 |
| 0.80  | +0.02 | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | +0.15 | 0.00  | 2/8 |
| 1.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0/8 |

Watch how the smallest-magnitude weights go to zero **first** and the largest hang on the longest. That's the sparsity pattern you'll see (approximately) in your own λ sweep.

---

## § 13 — What to expect from your λ sweep

You're going to run training with five values of $\lambda$: $\{0,\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1}\}$. Before you run anything, **predict the shapes of the curves**. If reality matches your prediction, you understand. If it doesn't, understanding is waiting on the other side.

Think of $\lambda$ as a dial from "no penalty" to "maximum penalty". Each plot shows what happens as you turn that dial from left to right:

### i. Non-zero weights vs λ

```text
non-zero weights
10000 ■────────
 8000          ■───
 5000               ■───
 2000                    ■───
  500                         ■
      ──────────────────────────── λ
      0    1e-4  1e-3  1e-2  1e-1
```

**Goes down.** More penalty → more weights killed. At $\lambda = 0$ all 10,000 weights are alive. At $\lambda = 0.1$ most are dead. This is L1 doing feature selection — the dial is choosing how many words matter.

### ii. Train metric vs λ

```text
train accuracy
 0.90 ■────────■───
 0.80                ■───
 0.70                     ■───
 0.55                          ■
      ──────────────────────────── λ
      0    1e-4  1e-3  1e-2  1e-1
```

**Goes down.** More penalty → model has less freedom → can't fit the training data as well. At $\lambda = 0$ it memorizes everything. At $\lambda = 0.1$ it's too constrained to learn much.

### iii. Validation metric vs λ — THE IMPORTANT ONE

```text
val accuracy
 0.78           ■───■
 0.75 ■────■              ■
 0.70                          ■
      ──────────────────────────── λ
      0    1e-4  1e-3  1e-2  1e-1
```

**U-shaped** (inverted-U for accuracy). This is the whole reason you're doing the sweep:

| λ | What happens | Val accuracy |
|---|---|---|
| 0 | No penalty. Model memorizes noise in training data. Overfits. | OK but not best |
| Small (1e-4, 1e-3) | Kills noisy features. Model focuses on useful words. Generalizes better. | **Best** |
| Big (1e-1) | Kills useful features too. Model can't learn anything. Underfits. | Bad |

The **sweet spot** — the peak of the curve — is the $\lambda$ you'd pick for production. It's the best trade-off between "fit the data" and "stay simple."

### iv. Individual weight trajectories

Pick 5 features that L1 eliminated (weights near zero at the end) and plot their weight value over training steps:

```text
weight value
 0.01 ──╲
         ╲──╲
              ╲──╲
                   ╲── → 0.0  (feature died around step 400)
      ──────────────────────── step
      0    200   400   600
```

You should see a clean decay toward zero. For features that **survive**, the weights bounce around but stay nontrivial. Plot both kinds on the same axes — the contrast tells the story.

### v. The one-sentence summary for your insights

### v. Example: what the results table might look like

Before running, this is the *rough shape* of the numbers you should expect (yours will differ — but the trends should match):

```text
λ        Non-zero weights   Train acc   Val acc    What happened
──────   ────────────────   ─────────   ─────────  ─────────────────────────
0        10000              0.90        0.75       overfits — memorizes noise
1e-4     9800               0.89        0.76       slight cleanup
1e-3     7500               0.86        0.78  ←── sweet spot (val peaks!)
1e-2     3000               0.78        0.76       more features killed
1e-1     200                0.58        0.55       model crippled — underfits
```

Look at the val accuracy column — it rises to a peak around $\lambda = 10^{-3}$ and then falls. That peak is where the model has learned to ignore noise without losing the ability to use useful features. **That is the trade-off you're documenting.**

### vi. The one-sentence summary for your insights

> ### *Small λ kills noise features and improves generalization. Large λ kills everything and collapses. The optimal λ is where validation accuracy peaks.*

---

## § 14 — The five tasks, distilled

Below is one card per sub-task. Each card lists the **concepts you'll need** from this guide and the **thinking** required to assemble the answer. None of these cards contain code — that's your part.

---

### Task 1.1 — Binary vs softmax & numerical stability
*10 points · conceptual + implementation*

**Concepts you need:**
- Sigmoid saturation in floating-point (§2)
- Why log(0) explodes (§4)
- Clamping with ε = 10⁻¹⁵ (§4)
- The max-subtraction trick & its proof (§5)
- Log-sum-exp form (§5)

**Thinking:**
- For the conceptual answers: explain in your own words what fails, why subtracting max preserves the result (algebra: factor out $e^{-c}$), and why every exponent ≤ 0 is now safe.
- For the BCE function: write the formula, but clamp `y_pred` before any `log` call.
- Verify on a stress test: pass logits like ±50 and confirm no NaN.

---

### Task 1.2 — The LogisticRegression class
*10 points · PyTorch nn.Module*

**Concepts you need:**
- `nn.Module` basics (§6)
- Why `w` must be shape `(n_features, 1)` (§6)
- `nn.Parameter` wrapping (§6)
- `.clone().detach()` hygiene (§6)
- Init options & "small" random scale (§7)

**Thinking:**
- In `__init__`: branch on the `init` argument, build the right tensor, wrap with `nn.Parameter`.
- Initialize bias as `nn.Parameter(torch.zeros(1))`.
- In `forward`: matrix multiply, add bias, sigmoid. Three lines.
- In `predict`: forward then threshold at 0.5 — return integer labels, not floats.

---

### Task 1.3 — Train with mini-batch SGD
*10 points · the training loop*

**Concepts you need:**
- The SGD loop pattern (§8)
- Shuffling matters (§8)
- `zero_grad` before `backward` (§8)
- Choosing & defending a metric (§10)

**Thinking:**
- Build the model and a `torch.optim.SGD` optimizer with the given `lr`.
- Each epoch: shuffle indices, iterate over them in chunks of size `batch_size`, run forward / loss / backward / step.
- After each batch, snapshot `w` and `b` into history (**detach + clone**, otherwise you'll save references that all point to the latest values).
- After each epoch, compute & log: train loss, train metric, val metric.
- For the metric: pick accuracy or F1; in your write-up, cite the class balance you observed.

---

### Task 1.4 — Hyperparameter sweep & heatmap
*20 points · experiments + analysis*

**Concepts you need:**
- How lr affects convergence vs stability (§9)
- How batch size affects gradient noise (§9)
- The lr × batch interaction & the diagonal ridge (§9)

**Thinking:**
- Two nested loops over the lr and batch_size lists. For each combo, call your `sgd_logistic_regression` and store the final train + val metric.
- Plot two heatmaps (train + val) with `matplotlib.imshow` or `seaborn.heatmap`. Annotate the cells with the values.
- In your analysis: identify the diagonal ridge, point out where divergence shows up (corner with small batch + large lr), point out underfitting (corner with large batch + small lr). **Tie each pattern to a specific cell of your heatmap.**

---

### Task 1.5 — L1 regularization & sparsity
*20 points · regularization + experiments*

**Concepts you need:**
- What regularization does (§11)
- L1 vs L2 geometric intuition (§12)
- Why SGD doesn't make exact zeros (§12)
- Predicted shapes for the four plots (§13)

**Thinking:**
- Modify your loss inside the training loop: when `penalty == 'l1'`, add `reg_lambda * torch.abs(model.w).sum()` to the BCE before `.backward()`. **Don't penalize bias.**
- For the init comparison: run the same setup with `init='zeros'` and `init='random'`. Record final metrics + count of weights with `|w| < 1e-7`. Discuss any difference.
- For the λ sweep: nested loop, collect train metric, val metric, non-zero count for each λ in {0, 1e-4, 1e-3, 1e-2, 1e-1}.
- Make four plots: non-zero count vs λ, train vs λ, val vs λ, individual weight trajectories for ~5 features L1 kills.
- Write the insights paragraph with reference to the U-shape in val and where you'd pick λ.

---

---

# Part 2 — Comparing Optimization Algorithms (30 pts)

> Part 1 trained logistic regression on a convex, well-conditioned loss. Everything worked because the problem was easy. **Part 2 throws optimizers at a non-convex, narrow-valley, many-local-minima landscape** — precisely so you can see where plain SGD *breaks* and why the other three were invented.

---

## § 15 — What an optimizer actually does

An optimizer is the rulebook that turns gradients into parameter updates. Plain SGD — the one you built in Part 1 — is the simplest possible rule: *take a small step in the direction of the negative gradient, repeat*. Every other optimizer you will ever meet — Momentum, AdaGrad, RMSprop, Adam, Lion, Shampoo — is a refinement of that same core idea, trying to make each step smarter in some specific way.

The scaffolding around every optimizer is always the same. On each step you:

1. Compute the gradient of the loss at the current parameters.
2. Read (and update) some internal state — possibly nothing, possibly a velocity vector, possibly a running average of squared gradients.
3. Combine the gradient and the state into an update direction and size.
4. Apply the update to the parameters.

The difference between GD, Momentum, AdaGrad, and Adam lives entirely in steps 2 and 3. The shell is identical. **Once you internalize that, all four optimizers stop being four separate recipes and start being four variations on a theme.**

> ### *Four optimizers, one loop, four update rules. The loop is the boring part.*

---

## § 16 — The four optimizers, in order

You will implement these in order of increasing cleverness. Each one fixes a specific weakness in the one before it. Understanding *why* each addition was made matters far more than memorizing the formula — the formulas follow naturally from the motivation.

### i. Gradient Descent — the baseline

$$\theta_{t+1} = \theta_t - \eta\, \nabla f(\theta_t)$$

No memory of past gradients. Each step is myopic — it looks only at the slope right here, right now, and marches in the steepest-descent direction.

In a nice round basin this is perfectly adequate. In a narrow curved valley, the gradient points *across* the valley toward the opposite wall, and plain GD bounces from wall to wall, making almost no progress along the valley floor. This is the classic "ravine" failure mode and it's exactly what the Six-hump Camel will expose.

### ii. Momentum — a heavy ball

$$v_{t+1} = \beta\, v_t + \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta\, v_{t+1}$$

Instead of a dimensionless point that teleports one step at a time, imagine a **heavy ball rolling downhill**. It has inertia. In a narrow valley where the gradient flips direction every few steps, the ball's accumulated velocity averages the oscillations out — side-to-side components cancel, forward motion along the valley floor survives. On a consistent slope the velocity keeps growing, so the ball actively *accelerates* downhill.

The hyperparameter $\beta$ (typically $0.9$) controls how much of the past velocity you carry forward. The "effective memory" of momentum is roughly $1/(1-\beta)$ steps — so $\beta = 0.9$ means you're averaging over the last ten gradients. Higher $\beta$ = more inertia = smoother path but slower turns.

> **Sidenote — Nesterov momentum.**
> A popular variant "looks ahead" by evaluating the gradient at $\theta_t - \eta\beta v_t$ instead of at $\theta_t$. You don't need it for this hometask, but most deep-learning code calling "momentum" is actually doing Nesterov by default.

### iii. AdaGrad — per-parameter learning rates

$$G_{t+1} = G_t + \nabla f(\theta_t) \odot \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1}} + \varepsilon} \odot \nabla f(\theta_t)$$

Momentum treats every parameter with the same learning rate. AdaGrad throws that out: it tracks a running sum of squared gradients for *each* parameter independently (the $\odot$ means elementwise), and divides each step by the square root of that sum. **Parameters that have had large gradients historically get damped; parameters that have been quiet get amplified.**

This is fantastic for sparse data — the classic example is a language model where common words get big gradients every batch and rare words get gradients once in a blue moon. AdaGrad automatically reins in the common words and lets the rare ones take bigger steps. You get per-feature learning rates for free.

**The fatal flaw:** $G$ only ever *grows*. It never decays. So the effective learning rate $\eta / \sqrt{G}$ monotonically shrinks — at roughly $O(t^{-1/2})$ — and eventually the optimizer stops moving even if the minimum is still far away. **AdaGrad "runs out of energy".** This is exactly the observation that motivated Adam.

### iv. Adam — the default for everything

$$m_{t+1} = \beta_1 m_t + (1 - \beta_1)\, \nabla f(\theta_t)$$
$$v_{t+1} = \beta_2 v_t + (1 - \beta_2)\, \nabla f(\theta_t)^2$$
$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{\,t+1}} \qquad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{\,t+1}}$$
$$\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \varepsilon}$$

Adam stands for "adaptive moment estimation", and it is cleaner than it looks. Read it as **Momentum + decaying AdaGrad + bias correction**:

- **$m$ is momentum, but decayed.** It is an exponential moving average of the gradient. Unlike classical momentum's ever-growing velocity, old contributions to $m$ slowly fade.
- **$v$ is AdaGrad, but forgiving.** It is an exponential moving average of the *squared* gradient. Unlike AdaGrad's monotonically-growing $G$, $v$ decays too — so the learning rate never dies. *This is the fix for AdaGrad's fatal flaw.*
- **The hats are bias correction.** Because $m$ and $v$ start at zero, the early values underestimate the true running averages (especially when $\beta_1, \beta_2$ are close to 1). Dividing by $(1 - \beta^{t+1})$ corrects for this startup bias. Without it, Adam's first few dozen steps would be unfairly small.
- **The final update divides momentum by RMS.** You move in the momentum direction, scaled per-parameter by the inverse root-mean-square of recent gradients — exactly what AdaGrad was doing, just with a finite memory window.

Typical defaults that you can copy without guilt: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, and learning rate $\eta$ somewhere from $10^{-4}$ to $10^{-2}$. The reason Adam is the default optimizer in almost every modern deep-learning codebase is not that it is always *best* — it's that it is almost never *catastrophic*. It works out-of-the-box on a shockingly wide variety of problems, including non-convex ones.

### Side-by-side cheat sheet

| Optimizer | State carried | Update rule (conceptual) | Best at | Worst at |
|---|---|---|---|---|
| **GD** | none | $\theta \leftarrow \theta - \eta g$ | simple convex | narrow valleys, ill-conditioning |
| **Momentum** | velocity $v$ | $v \leftarrow \beta v + g$;  $\theta \leftarrow \theta - \eta v$ | narrow valleys, consistent slopes | tuning $\beta$, overshoot |
| **AdaGrad** | sum $G$ of $g^2$ | $\theta \leftarrow \theta - \eta g / \sqrt{G}$ | sparse features, early training | late training (lr → 0) |
| **Adam** | EMA $m$ of $g$, EMA $v$ of $g^2$ | $\theta \leftarrow \theta - \eta \hat m / \sqrt{\hat v}$ | out-of-the-box everything | exact convex convergence |

---

## § 17 — Two landscapes — bowl & camel

Part 2 gives you two test functions, chosen specifically to expose the strengths and weaknesses of the four optimizers. Treat them as a **diagnostic**: the same optimizer will look very different on each, and the difference is what tells you what each method is actually for.

### A. The convex bowl

$$f(x,y) = x^2 + 2y^2$$

A single global minimum at the origin. The level sets are ellipses — *twice* as narrow in the $y$ direction as in the $x$ direction, because the $y$ coefficient is bigger. From any starting point, the gradient points roughly toward the origin; the only complication is that the $y$ direction is "steeper" than the $x$ direction, so vanilla GD with a single scalar learning rate will either overshoot in $y$ or creep in $x$. Watch how each optimizer handles this asymmetry — it's the simplest possible form of ill-conditioning, and it's already enough to separate the adaptive methods from GD.

### B. The Six-hump Camel

$$f(x,y) = \left(4 - 2.1 x^2 + \tfrac{x^4}{3}\right)x^2 + xy + \left(-4 + 4 y^2\right)y^2$$

A genuinely nasty playground. It has:

- **Two global minima** at $(+0.0898,\ -0.7126)$ and $(-0.0898,\ +0.7126)$, with value $\approx -1.0316$.
- **Four additional local minima** (the "six humps" in the name) plus several saddle points.
- **Narrow curved valleys** where vanilla GD oscillates wildly.
- **Flat-ish regions** where gradients are tiny and GD crawls.

You'll start from $(-2,\ -1.5)$ — deliberately far from any minimum, on the outskirts of the "interesting" region. Different optimizers will take *different paths* and may end up in *different basins*. Some will find a global minimum. Some will get trapped in a local one. **That outcome spread is the whole point.** Don't be alarmed if a method converges to a non-global minimum — document it, explain why, and you're doing the task correctly.

> **Sidenote — why this function?**
> The Six-hump Camel is a standard benchmark in global-optimization literature precisely because it combines smooth analytic structure with multiple local traps. It is hard enough to distinguish good methods from bad, but simple enough to visualize cleanly in 2D. Every optimization textbook uses it.

```text
Contour sketch of the camel (rough, not to scale)

   y
    │     . - ~ ~ - .
  2 ├   /             \        .─.  ← local min
    │  |    ◯           |     (   )
  1 ├  |   global        |     `─'
    │   \   min          /    ___
  0 ┼    ` .           . '   (   )  ← saddle
    │        global min       `─'
 -1 ├        ◯
    │     . - ~ ~ - .
 -2 ├ ★  START HERE              .─.
    │  (-2, -1.5)              (local min)
    └───┬──────┬──────┬──────┬──────┬── x
       -3     -2     -1      0      1      2
```

---

## § 18 — Reading your optimizer results

As in §13, the experiment is more useful if you **predict the shape of the answer before you run it.** Here's what to expect, roughly, and the reasoning behind each prediction.

### i. On the convex bowl

- **All four converge** to the origin. The bowl is convex — there is nowhere else to go.
- **GD is the slowest.** With a single scalar learning rate it cannot adapt to the elongated ellipses; pick too large an $\eta$ and it oscillates in $y$, pick too small and it crawls in $x$.
- **Momentum is notably faster.** Velocity accumulates along the long axis and the $y$-oscillations partially cancel. The trajectory looks smoother and more decisive.
- **AdaGrad handles the asymmetry automatically.** The $y$-gradient has historically been larger, so its effective learning rate is smaller. The path is more balanced than GD's.
- **Adam is typically fastest or tied.** On a problem this simple, any adaptive method will look clean. Expect a nearly-straight descent.

### ii. On the Six-hump Camel

- **GD is the most likely to get stuck.** Narrow valleys cause oscillations; local minima near the starting point capture it easily.
- **Momentum can escape shallow traps** by coasting through them on accumulated velocity. With the wrong $\eta$ or $\beta$, it overshoots past a good minimum and ends up somewhere else.
- **AdaGrad tends to "run out of steam"** — remember, its $G$ only grows, so after a while every step is tiny. It typically converges to *whatever* minimum is closest, global or not.
- **Adam usually finds a good minimum**, but the starting point matters. From $(-2, -1.5)$ you may or may not land in a global minimum depending on your lr. If you want to prove the basin-of-attraction story, try a handful of starting points.

### iii. Hyperparameter sensitivity

Do not expect one set of hyperparameters to work across both functions. The camel is genuinely harder and needs more care. Rough starting ranges to try:

- **GD:** $\eta \in [0.001,\ 0.1]$ for the bowl; smaller for the camel to avoid divergence.
- **Momentum:** same $\eta$ range as GD, with $\beta = 0.9$ as the standard.
- **AdaGrad:** $\eta \in [0.1,\ 1.0]$ — AdaGrad deliberately needs a bigger starting lr because it shrinks aggressively.
- **Adam:** $\eta \in [0.001,\ 0.1]$. The default of $10^{-3}$ is very conservative; on a toy 2-D problem you can often push it higher.

And give the experiment enough steps. The slower optimizers (plain GD in particular) need on the order of 2000 iterations to visibly converge on the bowl; less than that and you will unfairly penalise them.

### iv. The DRY hint

The hometask text asks you — sincerely — whether you can get rid of the repetition in the four optimizer functions. **You absolutely can.** Every optimizer follows the same loop; only the *update rule* changes. Factor out the loop, accept the update rule as a function parameter, and pass a different small function for each optimizer:

```text
def run(f, theta0, update_step, n_steps, **hparams):
    theta      = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    state      = {}                  # per-optimizer scratch (v, G, m, etc.)
    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for t in range(n_steps):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            update_step(theta, state, t, **hparams)
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values
```

Each optimizer becomes a five-line `update_step` function. That is the shape the instructor is hinting at — **seeing it is part of the lesson.**

> ### *The boilerplate is identical. The four optimizers are four update rules, not four loops.*

### v. The analysis paragraphs

The hometask asks for a 2–3 paragraph summary. Use the following prompts as scaffolding, with reference to your specific plots:

- Which optimizer wins on the bowl, and why? (Tie it to the elongation of the ellipses.)
- Which optimizer wins on the camel, and does it reliably find a global minimum from $(-2, -1.5)$? If not, which basin does it land in?
- Do the same hyperparameters transfer between the two functions? If you had to pick one lr per optimizer for both, what would it cost you?
- For each optimizer beyond plain GD, what specific weakness of GD does it fix, and what new weakness (if any) does it introduce?

A good answer cites specific cells of your plots — "on the camel, GD oscillates in the ravine near $(-1, 0.5)$ between steps 40 and 120" — rather than speaking in generalities.

---

## § 19 — The sixth task, distilled

Same structure as the Part 1 cards in §14. Concepts on the left, thinking on the right, no solutions anywhere.

### Task 2 — Optimizers on simple vs. hard functions
*30 points · implementation + experiments + analysis*

**Concepts you need:**
- The shared optimizer scaffolding (§15)
- GD, Momentum, AdaGrad, Adam update rules (§16)
- Why Momentum helps in narrow valleys (§16)
- AdaGrad's shrinking-lr fatal flaw (§16)
- Adam = Momentum + decaying AdaGrad + bias correction (§16)
- Convex bowl vs Six-hump Camel landscape (§17)
- Expected behavior of each optimizer on each (§18)
- The DRY scaffolding hint (§18)

**Thinking:**
- Fill in the four skeletons: compute loss, call `backward`, update parameters per the rule, zero the grad.
- For **Momentum**: the velocity `v` must persist across iterations — declare it outside the inner loop.
- For **AdaGrad**: same, but with elementwise squared-gradient accumulation into `G`. Don't forget the `+ eps` in the denominator.
- For **Adam**: two state tensors (`m`, `v`), plus bias correction using the current step index `t` (note: `t+1` for the exponent if you're zero-indexed).
- Run each optimizer on **both** functions from $(-2,\ -1.5)$ with at least 2000 steps.
- One plot of function value vs iteration *per function*, with all four optimizers overlaid on the same axes.
- Trajectory overlays on the contour plot of each function (use `plot_trajectories_camel_log`).
- Write the analysis paragraphs answering the four prompts in §18.v. **Cite specific plot regions.**
- **Bonus** (encouraged by the prompt): collapse the four near-duplicate functions into one with a pluggable update rule. See the scaffolding in §18.iv.

---

## Colophon

*Hand-bound for one student, on one weekend, before one homework deadline.*

A companion HTML version of this guide lives at `02b/logistic_regression_guide.html` with two interactive playgrounds (a sigmoid + BCE slider and a live L1 soft-thresholding demo). This markdown version is the portable, offline-friendly sibling — and now covers all 100 points of Hometask 01.

— Field Guide № 01 · Logistic Regression &amp; Optimization —
