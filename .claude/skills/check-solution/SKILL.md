---
name: check-solution
description: Review code from the notebook and check for correctness, bugs, and missing pieces. Use when the user shares code and asks "is this correct?", "does this look right?", "check my solution", "do I have an error?".
user-invocable: true
---

# Check Solution

Review the user's code (from their message or selected lines) and provide a thorough correctness check.

## Process

1. **Read the code** — Understand what the code is trying to do based on the task context.

2. **Check for correctness** — Verify:
   - Does the logic match the mathematical formula?
   - Are tensor shapes correct throughout?
   - Is the order of operations right (especially zero_grad / backward / step)?
   - Are there off-by-one errors?
   - Is numerical stability handled (clamping, epsilon)?

3. **Check for common bugs** — Look for:
   - Missing `.detach().clone()` when saving history
   - Missing `torch.no_grad()` around evaluation code
   - Missing `+ 1e-9` in division (precision, recall, F1)
   - Wrong order of `zero_grad`, `backward`, `step`
   - Missing `optimizer.step()`
   - Shape mismatches that PyTorch silently broadcasts
   - Using `model.w` instead of `nn.Parameter(model.w)`

4. **Report** — Present findings as:
   - A line-by-line verdict table (line | what it does | correct?)
   - Any bugs found with the exact fix needed
   - Confirmation of what IS correct (don't only focus on errors)

## Rules

- Be specific: point to the exact line and variable.
- If everything is correct, say so clearly — don't invent problems.
- Don't rewrite the user's code unless there's a bug. If the style is different but correct, leave it.
- Always explain WHY something is wrong, not just that it is.
