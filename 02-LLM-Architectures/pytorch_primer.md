# PyTorch Primer — Just What You Need for Hometask 01

> A practical reference for every PyTorch concept used in the hometask. Each concept has 2–3 examples and a note about where you'll use it. Keep this open while you code.

**Mental model:** PyTorch = NumPy + autograd. For this hometask, we only care about two things:

1. **Tensors** — n-dimensional arrays you can do math on (like NumPy arrays).
2. **Autograd** — the magic that computes derivatives automatically so you don't have to.

Everything else is thin wrapping around those two ideas.

---

## Table of contents

1. [Tensors — the fundamentals](#1-tensors--the-fundamentals)
2. [Creating tensors](#2-creating-tensors)
3. [Math operations you'll need](#3-math-operations-youll-need)
4. [Autograd — how `.backward()` works](#4-autograd--how-backward-works)
5. [Cloning, detaching & `no_grad`](#5-cloning-detaching--no_grad)
6. [`nn.Module` & `nn.Parameter`](#6-nnmodule--nnparameter)
7. [Optimizers](#7-optimizers)
8. [Shuffling & indexing](#8-shuffling--indexing)
9. [`requires_grad` for Task 2](#9-requires_grad-for-task-2)
10. [Cheat sheet](#cheat-sheet)
11. [Common mistakes](#common-mistakes-to-avoid)

---

## 1. Tensors — the fundamentals

**What it is:** A tensor is PyTorch's version of a NumPy array — an n-dimensional grid of numbers. The key difference: tensors can track gradients and run on a GPU.

### Example 1 — Create from a Python list

```python
import torch

t = torch.tensor([1.0, 2.0, 3.0])
print(t)           # tensor([1., 2., 3.])
print(t.shape)     # torch.Size([3])
print(t.dtype)     # torch.float32
```

### Example 2 — Create from a NumPy array (what you'll actually do)

```python
import numpy as np
import torch

X = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.tensor(X, dtype=torch.float32)
print(t.shape)     # torch.Size([2, 3])
```

The `dtype=torch.float32` is important. If you don't specify it, a NumPy integer array becomes an integer tensor, and you can't do gradient-based math on integers.

### Example 3 — Reshape with `.view()`

```python
y = torch.tensor([0, 1, 0, 1])     # shape (4,)      — a flat vector
y = y.view(-1, 1)                   # shape (4, 1)   — a column

print(y)
# tensor([[0],
#         [1],
#         [0],
#         [1]])
```

The `-1` means "figure out this dimension automatically". You'll use this in Task 1.3 to reshape labels from `(n,)` to `(n, 1)` so they match the sigmoid output shape.

**Where you use it:** every line of the hometask.

---

## 2. Creating tensors

Three factory functions you need:

### `torch.zeros` — all zeros, given a shape

```python
z = torch.zeros(3, 2)
# tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.]])

# For logistic regression weights (10000 features, 1 output):
w = torch.zeros(10000, 1)
print(w.shape)   # torch.Size([10000, 1])
```

### `torch.randn` — standard normal random values

```python
r = torch.randn(3, 2)          # mean 0, std 1
# tensor([[ 0.54, -1.23],
#         [ 0.11,  2.05],
#         [-0.89,  0.42]])

small_r = torch.randn(3, 2) * 0.01   # scaled down for "small random init"
# tensor([[ 0.005, -0.012],
#         [ 0.001,  0.020],
#         [-0.008,  0.004]])
```

The `* 0.01` scaling is **critical** for logistic regression. Without it, early sigmoid outputs saturate and training crawls. This is the "small random values" the assignment mentions.

### `torch.zeros_like` — same shape as another tensor

```python
theta = torch.tensor([1.0, 2.0])
v = torch.zeros_like(theta)
print(v)          # tensor([0., 0.])
print(v.shape)    # torch.Size([2])
```

You'll use this in Task 2 to initialize velocity (for Momentum) and G accumulator (for AdaGrad) — they must match the shape of `theta`.

**Where you use it:** Task 1.2 (weight init), Task 2 (optimizer state init).

---

## 3. Math operations you'll need

### Matrix multiplication — the `@` operator

```python
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])    # shape (2, 3)

w = torch.tensor([[0.5],
                  [1.0],
                  [1.5]])            # shape (3, 1)

result = x @ w                       # shape (2, 1)
# tensor([[ 7.0],      # 1*0.5 + 2*1.0 + 3*1.5
#         [16.0]])     # 4*0.5 + 5*1.0 + 6*1.5
```

```python
# Task 1.2 forward pass:
logits = x @ self.w + self.b
#        (batch, n_features) @ (n_features, 1)  →  (batch, 1)
```

### `torch.sigmoid` — squash real numbers into (0, 1)

```python
z = torch.tensor([-2.0, 0.0, 2.0])
probs = torch.sigmoid(z)
# tensor([0.1192, 0.5000, 0.8808])
```

```python
# Inside forward():
probs = torch.sigmoid(logits)
```

### `torch.log` — natural logarithm

```python
p = torch.tensor([0.5, 0.9, 0.1])
torch.log(p)
# tensor([-0.6931, -0.1054, -2.3026])

# The danger:
torch.log(torch.tensor([0.0]))
# tensor([-inf])   ← this kills training
```

### `torch.clamp` — force values into a range

```python
x = torch.tensor([-5.0, 0.3, 99.0])
torch.clamp(x, 0.0, 1.0)
# tensor([0.0000, 0.3000, 1.0000])
```

```python
# Task 1.1 — save the log from seeing 0 or 1:
y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
```

### `torch.abs` — absolute value

```python
w = torch.tensor([-1.5, 0.3, -2.0])
torch.abs(w)
# tensor([1.5000, 0.3000, 2.0000])

# Sum for L1 penalty:
l1_penalty = torch.abs(w).sum()   # tensor(3.8)
```

### `.mean()` and `.sum()` — reduce a tensor to a scalar

```python
loss_per_example = torch.tensor([0.5, 0.8, 0.2, 0.1])
loss = loss_per_example.mean()    # tensor(0.4)
total = loss_per_example.sum()    # tensor(1.6)
```

**Critical rule:** the loss passed to `.backward()` must be a **scalar**. Always `.mean()` or `.sum()` before calling backward.

**Where you use it:**
- `@` → every forward pass
- `sigmoid` → Task 1.2
- `log`, `clamp` → Task 1.1 (BCE)
- `abs` → Task 1.5 (L1 penalty)
- `mean` → every loss computation

---

## 4. Autograd — how `.backward()` works

**What it is:** If you mark a tensor with `requires_grad=True`, PyTorch silently builds a computation graph as you operate on it. When you call `.backward()` on a scalar, it walks the graph backwards and fills in `.grad` on every tracked tensor.

### Example 1 — The simplest case

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 5        # y = 14
y.backward()          # compute dy/dx
print(x.grad)         # tensor(6.)   ← because dy/dx = 2x = 6 when x = 3
```

### Example 2 — With a vector

```python
theta = torch.tensor([2.0, 3.0], requires_grad=True)
loss = (theta ** 2).sum()    # scalar: 4 + 9 = 13
loss.backward()
print(theta.grad)            # tensor([4., 6.])   ← d(x²)/dx = 2x, elementwise
```

### Example 3 — Gradients accumulate, you MUST zero them

```python
theta = torch.tensor(1.0, requires_grad=True)

loss1 = theta * 2
loss1.backward()
print(theta.grad)     # tensor(2.)

loss2 = theta * 3
loss2.backward()
print(theta.grad)     # tensor(5.)   ← 2 + 3, NOT 3!

theta.grad.zero_()    # reset to zero
loss3 = theta * 4
loss3.backward()
print(theta.grad)     # tensor(4.)   ← correct now
```

**This is why `optimizer.zero_grad()` is called before every batch.** Without it, the gradient from the previous batch leaks into the current one and training explodes.

**Where you use it:** Task 1.3 training loop, Task 2 all four optimizers.

---

## 5. Cloning, detaching & `no_grad`

These three work together to let you "step out of" autograd when you don't want it involved.

### `torch.no_grad()` — "don't track anything inside this block"

```python
with torch.no_grad():
    val_probs = model(x_val)
    val_preds = (val_probs > 0.5).float()
    accuracy = (val_preds == y_val).float().mean()
```

Why: evaluation doesn't need gradients. `no_grad` makes it faster and uses less memory. **Always wrap your validation code in `no_grad`.**

### `.detach()` — return a version not tracked by autograd

```python
theta = torch.tensor(3.0, requires_grad=True)
y = theta ** 2

y_frozen = y.detach()    # same value (9), but disconnected from the graph
# y_frozen.backward()    # would error — no graph attached
```

You use this when you want the **value** of a tensor but you don't want anything you do with it to affect training.

### `.clone()` — make an independent copy

```python
a = torch.tensor([1.0, 2.0])
b = a                # b is a REFERENCE to a — not a copy
b[0] = 99
print(a)             # tensor([99., 2.])   ← a also changed!

a = torch.tensor([1.0, 2.0])
b = a.clone()        # b is a real copy
b[0] = 99
print(a)             # tensor([1., 2.])    ← a untouched
```

### Putting it together — saving training history

```python
history = []
for batch in batches:
    # ... training step ...
    history.append({
        'w': model.w.detach().clone(),
        'b': model.b.detach().clone(),
    })
```

- `.detach()` → don't bring the computation graph along (wastes memory and confuses things)
- `.clone()` → don't share memory with `model.w`, which is about to change on the next step

**If you forget `.clone()`**, every entry in `history` will end up pointing to the SAME tensor — the final one — because you saved references, not values. This is one of the most common PyTorch mistakes.

### Manual parameter update inside `no_grad` (Task 2)

```python
with torch.no_grad():
    theta -= lr * theta.grad     # mutate theta in place
```

You need `no_grad` here because you're modifying a tracked tensor in place. Without it, autograd gets confused.

**Where you use it:**
- `no_grad` → validation (Task 1.3) and manual updates (Task 2)
- `.detach().clone()` → history snapshots (Task 1.3)

---

## 6. `nn.Module` & `nn.Parameter`

`nn.Module` is PyTorch's base class for "anything with parameters and a forward pass". You subclass it, and inside you wrap learnable tensors in `nn.Parameter` so PyTorch knows to track them.

### Example 1 — A minimal model

```python
import torch
import torch.nn as nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()              # don't forget this line
        self.w = nn.Parameter(torch.zeros(3, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x @ self.w + self.b

model = Tiny()
x = torch.tensor([[1., 2., 3.]])
print(model(x))
# tensor([[0.]], grad_fn=<AddBackward0>)
```

Notice the `grad_fn` — it means the output is tracked. That's `nn.Parameter` doing its job.

### Example 2 — Accessing parameters

```python
model = Tiny()

for name, param in model.named_parameters():
    print(name, param.shape)
# w torch.Size([3, 1])
# b torch.Size([1])

# This is what the optimizer uses:
list(model.parameters())
# [Parameter containing: tensor([...], requires_grad=True),
#  Parameter containing: tensor([...], requires_grad=True)]
```

### Example 3 — Why you MUST use `nn.Parameter`

```python
# BROKEN — plain tensor isn't tracked
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.zeros(3, 1)   # missing nn.Parameter!

model = Broken()
print(list(model.parameters()))   # []  — empty!
```

Without `nn.Parameter`, `model.parameters()` returns nothing, the optimizer has no weights to update, and training silently does nothing. This is a nasty bug because it fails silently.

**Where you use it:** Task 1.2 — the entire `LogisticRegression` class.

---

## 7. Optimizers

An optimizer is an object that holds a reference to your parameters and knows how to update them. You don't write `w -= lr * w.grad` by hand — the optimizer does it.

### Example 1 — SGD basics

```python
import torch.optim as optim

model = LogisticRegression(n_features=10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

`model.parameters()` tells the optimizer which tensors to update.

### Example 2 — The three-step training dance

```python
for batch in batches:
    optimizer.zero_grad()      # 1. clear old gradients
    y_hat = model(x_batch)
    loss  = bce(y_hat, y_batch)
    loss.backward()            # 2. compute new gradients
    optimizer.step()           # 3. apply w -= lr * w.grad for every param
```

Every training step does these three in this exact order:

| Step | What happens | If you forget |
|---|---|---|
| `zero_grad()` | Clears `.grad` to 0 | Gradients accumulate → training explodes |
| `backward()` | Populates `.grad` | Optimizer updates with stale gradients |
| `step()` | Updates the weights | Weights never change, loss never drops |

### Example 3 — Momentum is a one-liner

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

You won't need this for Part 1 (plain SGD is fine), but it shows that `torch.optim.SGD` already supports momentum out of the box. In Task 2 you'll implement momentum yourself — but only because the exercise asks you to.

**Where you use it:** Task 1.3 and 1.5.

---

## 8. Shuffling & indexing

### `torch.randperm` — a random permutation of `[0, 1, ..., n-1]`

```python
perm = torch.randperm(10)
# tensor([3, 7, 1, 9, 0, 5, 8, 2, 4, 6])   ← different each run
```

### Example 1 — Shuffle a dataset before iterating in batches

```python
n = 6920
perm = torch.randperm(n)

X_shuffled = X_train[perm]
y_shuffled = y_train[perm]

for i in range(0, n, 100):
    x_batch = X_shuffled[i:i+100]
    y_batch = y_shuffled[i:i+100]
    # ... training step ...
```

When `i + 100 > n`, the slice `[i:i+100]` returns whatever's left (a smaller final batch). That's fine — PyTorch handles variable-size batches automatically.

### Example 2 — Fancy indexing with a tensor of indices

```python
x = torch.tensor([[10, 20],
                  [30, 40],
                  [50, 60],
                  [70, 80]])

idx = torch.tensor([0, 2, 3])
print(x[idx])
# tensor([[10, 20],
#         [50, 60],
#         [70, 80]])
```

This is why `X_train[perm]` works — `perm` is a tensor of row indices, and PyTorch returns the rows in that order.

### Example 3 — Reshuffle every epoch

```python
for epoch in range(num_epochs):
    perm = torch.randperm(n)         # new shuffle each epoch
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    for i in range(0, n, batch_size):
        # ... training step ...
```

**Where you use it:** Task 1.3 training loop.

---

## 9. `requires_grad` for Task 2

For Part 1 you hide gradient tracking inside `nn.Module`. For Part 2 you're working directly on bare tensors, so you need to request gradient tracking explicitly.

### Example 1 — Create a trainable tensor from scratch

```python
theta = torch.tensor([-2.0, -1.5], dtype=torch.float32, requires_grad=True)
# theta is a "free-floating" parameter — no nn.Module needed
```

### Example 2 — The full manual loop for plain GD

```python
theta = torch.tensor([-2.0, -1.5], dtype=torch.float32, requires_grad=True)
lr = 0.01

for step in range(2000):
    loss = camel(theta)             # your function
    loss.backward()                  # populates theta.grad

    with torch.no_grad():
        theta -= lr * theta.grad     # manual update

    theta.grad.zero_()               # manual zero
```

### Example 3 — Momentum, by hand

```python
theta = torch.tensor([-2.0, -1.5], dtype=torch.float32, requires_grad=True)
v = torch.zeros_like(theta)     # velocity state
lr = 0.01
beta = 0.9

for step in range(2000):
    loss = camel(theta)
    loss.backward()

    with torch.no_grad():
        v = beta * v + theta.grad   # update velocity
        theta -= lr * v              # step

    theta.grad.zero_()
```

Notice: no `torch.optim`, no `nn.Module`, just a tensor and a loop. That's the whole Task 2.

**Where you use it:** Task 2 — all four optimizers.

---

## Cheat sheet

Print this. Keep it next to the screen.

| Operation | One-liner |
|---|---|
| Create float tensor from numpy | `torch.tensor(X, dtype=torch.float32)` |
| Reshape to column | `t.view(-1, 1)` |
| Zeros / random small | `torch.zeros(n, 1)` / `torch.randn(n, 1) * 0.01` |
| Same-shape zeros | `torch.zeros_like(theta)` |
| Matrix multiply | `x @ w` |
| Add bias | `x @ w + b` |
| Sigmoid | `torch.sigmoid(z)` |
| Clamp for BCE | `torch.clamp(p, 1e-15, 1 - 1e-15)` |
| Natural log | `torch.log(p)` |
| Mean / sum reduce | `t.mean()` / `t.sum()` |
| L1 penalty | `torch.abs(w).sum()` |
| Track gradients | `requires_grad=True` |
| Compute gradients | `loss.backward()` |
| Read gradient | `t.grad` |
| Zero gradient | `t.grad.zero_()` |
| No-grad context | `with torch.no_grad():` |
| Detach + copy | `t.detach().clone()` |
| Shuffle indices | `torch.randperm(n)` |
| SGD optimizer | `optim.SGD(model.parameters(), lr=0.01)` |
| Training three-step | `zero_grad()` → `backward()` → `step()` |
| Wrap learnable tensor | `nn.Parameter(tensor)` |

---

## Common mistakes to avoid

1. **Forgetting `.mean()` or `.sum()` on the loss** → loss isn't a scalar → `.backward()` fails with a cryptic error.
2. **Forgetting `optimizer.zero_grad()`** → gradients accumulate across batches → loss explodes within a few steps.
3. **Saving `model.w` instead of `model.w.detach().clone()`** → every entry in your history list points to the same final tensor. Debugging this wastes an hour.
4. **Using `torch.tensor(np_array)` without `dtype=torch.float32`** → you get a float64 tensor that may mismatch the float32 model weights → silent performance cliff or crash.
5. **Plain tensor instead of `nn.Parameter` inside a Module** → `model.parameters()` is empty → the optimizer updates nothing → loss stays flat forever.
6. **Forgetting `requires_grad=True` for Task 2** → `theta.grad` is `None` → `.backward()` has nothing to populate.
7. **Mutating a tracked tensor outside `no_grad`** → autograd complains with "a leaf Variable that requires grad is being used in an in-place operation."
8. **Reusing `theta.grad` after `.backward()` without zeroing** → same as #2, just for Task 2 instead of Task 1.3.

---

## Minimum working examples — copy these mental models

### Task 1.1 BCE (you already have this)

```python
y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
return loss.mean()
```

### Task 1.2 nn.Module skeleton

```python
class LogisticRegression(nn.Module):
    def __init__(self, n_features, init="zeros"):
        super().__init__()
        if init == "zeros":
            w = torch.zeros(n_features, 1)
        elif init == "random":
            w = torch.randn(n_features, 1) * 0.01
        elif isinstance(init, torch.Tensor):
            w = init.clone().detach()
        else:
            raise ValueError(...)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ...      # logits, then sigmoid, return probs
```

### Task 1.3 training loop skeleton

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(epochs):
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        x_b = X_train_tensor[idx]
        y_b = y_train_tensor[idx]

        optimizer.zero_grad()
        y_hat = model(x_b)
        loss = binary_cross_entropy_loss(y_hat, y_b)
        loss.backward()
        optimizer.step()

        history.append({
            'w': model.w.detach().clone(),
            'b': model.b.detach().clone(),
        })
```

### Task 2 manual GD

```python
def gradient_descent(f, theta0, lr=0.001, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            theta -= lr * theta.grad
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values
```

---

*That's everything. The total API surface for this hometask is about 15 functions. If you understand all of the concepts in this primer, you can complete 100/100 points.*

— PyTorch Primer · Companion to the Logistic Regression Field Guide —
