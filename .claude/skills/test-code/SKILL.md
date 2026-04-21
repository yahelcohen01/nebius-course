---
name: test-code
description: Generate a sanity-check test block that validates a function or class without revealing the solution. Use when the user says "test my code", "give me tests", "how can I verify", "sanity check".
argument-hint: [function-or-class-name]
user-invocable: true
---

# Generate Test Code

Create a test block for: **$ARGUMENTS**

## Process

1. **Understand what the function/class should do** — Read the assignment to understand the expected behavior, inputs, outputs, and edge cases.

2. **Write assertion-based tests** that check:
   - Output shapes are correct
   - Output dtypes are correct
   - Output values are in expected ranges
   - Edge cases are handled (empty input, zero weights, etc.)
   - Known input-output pairs (e.g., sigmoid(0) = 0.5)
   - Parameters are properly registered (for nn.Module classes)
   - Gradients flow (backward() populates .grad)
   - No shared memory between snapshots (.clone().detach() hygiene)

3. **Print clear pass/fail messages** for each test:
   ```python
   assert condition, "error message"
   print("Test N: description — passed")
   ```

4. **End with a summary line:**
   ```python
   print("\nAll tests passed. Your implementation looks correct.")
   ```

## Rules

- Tests must NOT reveal the solution. Test behavior and properties, not implementation details.
- Use small, handcrafted inputs where the expected output can be computed by hand.
- Each test should be independent — failing one shouldn't block others (use try/except if needed).
- Include a "what each test checks" table after the code block.
- Include a "if test N fails, common causes" section.
- The test block should be copy-pasteable into a notebook cell.
