# AI Course — Nebius · LLM Architectures

## Project

This is a university-level AI/ML course (Nebius). Each hometask is a Jupyter notebook (`.ipynb`) with tasks, point values, skeleton code, and TODO markers. Assignments live in numbered folders (`01a/`, `02b/`, etc.).

## Student Profile

- Learning ML — not a researcher. Explain concepts from the ground up with concrete examples.
- Wants to **understand, then solve alone**. Never give complete code solutions for hometask questions unless explicitly asked.
- Responds well to: small numeric examples first → scale to real size, physical analogies, step-by-step traces with actual numbers, side-by-side comparison tables.
- Reads guide files as raw markdown in the editor — avoid LaTeX in chat responses. Use `x²` not `$x^2$`, `w × x + b` not `$wx + b$`.
- LaTeX is fine inside `.html` and `.md` guide files (they get rendered by KaTeX / VS Code preview).

## Workflow for Each Hometask

Follow this sequence when starting a new assignment:

```
1. /read-hometask [notebook path]          → extract all tasks, points, requirements
2. /study-guide [topic]                    → create HTML + MD teaching guides
3. Student reads the guide and writes code
4. /check-solution                         → review their code for bugs
5. /test-code [function-name]              → generate sanity-check tests
6. Student runs experiments
7. /write-analysis                         → write analysis paragraphs from results
```

## Available Skills

| Command | Purpose |
|---------|---------|
| `/study-guide [topic]` | Research topic, create paired HTML + MD guides with interactive elements |
| `/read-hometask [file]` | Extract all tasks, points, deliverables from a notebook |
| `/check-solution` | Review code for correctness, bugs, missing pieces |
| `/explain [concept]` | Plain-language explanation with examples and analogies |
| `/test-code [name]` | Generate test blocks that validate without revealing solutions |
| `/write-analysis` | Write graded-assignment-quality analysis paragraphs from experiment results |
| `/pytorch-help [function]` | Quick 2-3 example reference for a PyTorch function |

## Study Guide Style

When creating guides (`/study-guide`):

- **HTML version**: warm cream paper aesthetic, Fraunces + Newsreader fonts, KaTeX math, interactive playgrounds (sliders, demos), hand-tuned SVG diagrams, section numbering with `§`, drop caps, sidenotes, pull quotes, task cards at the end
- **MD version**: same content adapted for markdown, ASCII diagrams, tables replacing interactives
- Both saved in the assignment folder (e.g., `02b/logistic_regression_guide.html`)
- **Teach, don't solve.** Task cards list concepts + thinking, never code implementations

## File Structure

```
nebius-course/
├── CLAUDE.md                           ← you are here
├── .claude/
│   └── skills/                         ← custom slash commands
│       ├── study-guide/SKILL.md
│       ├── read-hometask/SKILL.md
│       ├── check-solution/SKILL.md
│       ├── explain/SKILL.md
│       ├── test-code/SKILL.md
│       ├── write-analysis/SKILL.md
│       └── pytorch-help/SKILL.md
├── 01-ai-model-to-ai-product/
│   └── assignment_01/                           ← first assignment
├── 02-LLM-Architectures/
│   └── assignment_01/
│       ├── LLM_Architectures,_hometask_1.ipynb  ← hometask notebook
│       ├── logistic_regression_guide.html       ← HTML study guide
│       ├── logistic_regression_guide.md         ← MD study guide
│       ├── pytorch_primer.md                    ← PyTorch API reference
│       └── task2_solution.py                    ← optimizer comparison code
└── [future folders: 03-topic-name/, etc.]
```

## Code Conventions

- **Framework**: PyTorch (no TensorFlow/Keras)
- **Environment**: Jupyter notebooks, local Python with venv
- **Plotting**: matplotlib + seaborn
- **Metrics**: sklearn or hand-rolled from confusion matrix
- **No GPU** — everything runs on CPU for this course

## When Checking Solutions

- Point to the exact line and variable when reporting bugs.
- Confirm what IS correct — don't only focus on errors.
- The most common bugs: wrong order of `zero_grad`/`backward`/`step`, missing `.detach().clone()` for history, missing `+ 1e-9` in divisions, shape mismatches from broadcasting.

## When Explaining Concepts

- Start with a one-sentence definition, no jargon.
- Give the smallest possible example (3-5 elements) with every step shown.
- Then connect to the real assignment scale ("now imagine 10,000 words instead of 5").
- Use analogies: heavy ball for momentum, volume dial for lambda, surprise for cross-entropy.
- Always address the "why" — what problem does this solve, what goes wrong without it.

## When Writing Analysis

- Always cite specific numbers from the student's actual results.
- Never say "accuracy improved" — say "accuracy improved from 0.72 to 0.78".
- Tie observations back to theory from the study guide.
- Write in first person, conversational academic tone.
- Keep it to 2-4 paragraphs unless the assignment asks for more.
