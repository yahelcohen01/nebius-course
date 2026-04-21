---
name: pytorch-help
description: Quick PyTorch reference with 2-3 examples for a specific function or concept. Use when the user asks "how do I use torch.X", "what does torch.X do", "pytorch help", "show me how to use".
argument-hint: [function-or-concept]
user-invocable: true
---

# PyTorch Quick Reference

Explain and demonstrate: **$ARGUMENTS**

## Format

For each function or concept, provide:

### 1. One-line description
What it does in plain English.

### 2. Signature
```python
torch.function_name(param1, param2, ...)
```

### 3. Example 1 — The simplest case
```python
# input
x = torch.tensor([1.0, 2.0, 3.0])
# call
result = torch.function_name(x)
# output
# tensor([...])
```

### 4. Example 2 — With a 2D tensor (matrix)
Show how the function works on batched data, which is what the assignment actually uses.

### 5. Example 3 — In the assignment context
Show where this function appears in the hometask code (forward pass, training loop, evaluation, etc.) with a brief note about why it's used there.

### 6. Common mistakes
One or two things that go wrong when using this function.

## Rules

- Show input AND output for every example.
- Use small tensors (3-5 elements) so the user can verify by hand.
- Mention the shape of every tensor: `# shape (3,)` or `# shape (batch, 1)`.
- If the function has a gotcha (e.g., `torch.tensor` defaults to int for int input), flag it.
- If there's a related function the user might confuse it with, briefly note the difference.
- Keep it focused — one function per invocation. Don't turn it into a tutorial.
