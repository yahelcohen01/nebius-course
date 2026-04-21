---
name: write-analysis
description: Given experiment results (numbers, plots, tables), write analysis paragraphs suitable for a graded ML assignment. Use when the user says "write the analysis", "write the findings", "what should I write", "give me the paragraphs".
user-invocable: true
---

# Write Analysis

Write analysis paragraphs based on the experiment results the user has shared.

## Process

1. **Read the results** — Look at the numbers, tables, and plot descriptions the user provided.

2. **Identify the key patterns:**
   - What increased / decreased / stayed flat?
   - Where is the best result? The worst?
   - Any unexpected results? Any that confirm predictions?
   - Comparisons between methods / settings

3. **Write the paragraphs** following this structure:
   - **State the observation** — what the data shows, citing specific numbers.
   - **Explain why** — connect the observation to the theory (e.g., "this is because L1's constant gradient force...").
   - **Draw the conclusion** — what this means for practice (e.g., "the optimal lambda is 1e-3").

4. **Cite specific numbers from the results.** Never say "accuracy improved" — say "accuracy improved from 0.72 to 0.78". Reference specific cells, epochs, or data points.

## Rules

- Write in the user's voice — first person, conversational academic tone.
- Keep it concise: 2-4 paragraphs unless the assignment asks for more.
- Always tie observations back to theory from the study guide.
- Don't fabricate numbers — only use data the user actually provided.
- If the results are surprising or contradict expectations, acknowledge it and explain why.
- End with the practical takeaway: "the best setting is X because Y".
- The text should be ready to paste into the notebook as a markdown cell.
