---
name: explain
description: Explain an ML or programming concept simply with concrete examples, step-by-step traces, and analogies. Use when the user says "explain", "what is", "what does X mean", "I don't understand", "break this down".
argument-hint: [concept]
user-invocable: true
---

# Explain Concept

Explain **$ARGUMENTS** in a way that builds understanding from the ground up.

## How to explain

1. **Start with a one-sentence plain-language definition.** No jargon. If you must use a technical term, define it immediately.

2. **Give the simplest possible example first.** Use tiny numbers (3-5 elements), not real-world scale. Show every step of the computation explicitly.

3. **Then scale up.** Connect the tiny example to the real assignment context (e.g., "now imagine 10,000 words instead of 5").

4. **Use analogies** the user can relate to. Physical analogies work well (heavy ball for momentum, volume dial for lambda, etc.).

5. **Show the math with numbers, not just symbols.** Instead of just writing the formula, plug in concrete values and trace through step by step:
   ```
   formula: result = a × b + c
   numbers: result = 3 × 4 + 2 = 14
   ```

6. **Address the "why"** — not just what it does, but why it was designed this way. What problem does it solve? What goes wrong without it?

7. **Use tables for comparisons.** When comparing two things (L1 vs L2, accuracy vs F1, etc.), a side-by-side table is clearer than prose.

## Rules

- NO LaTeX in chat responses (the user reads raw text, not rendered math). Use plain symbols: `x²` not `$x^2$`, `w × x + b` not `$wx + b$`.
- Use code blocks for step-by-step traces.
- Keep it concrete. Every abstract statement should be followed by "for example..." with real numbers.
- Match the user's level — they are a student learning ML, not a researcher. Avoid assuming prior knowledge.
- If the concept relates to code they'll write, show what the code DOES (with a trace), not the code itself.
