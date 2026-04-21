---
name: study-guide
description: Research a topic and create paired HTML + MD study guides with interactive elements. Use when the user needs to learn a new topic for an assignment or hometask. Triggers on "create a guide", "teach me", "I need to learn", "study guide for".
argument-hint: [topic]
user-invocable: true
---

# Study Guide Creator

Create a comprehensive study guide for the topic: **$ARGUMENTS**

## Process

1. **Read the assignment** — Find and read the relevant notebook or assignment file to understand exactly what questions need to be answered and what concepts are required.

2. **Research the topic** — Use WebSearch to gather authoritative information on the topic. Focus on:
   - Core theory and intuition
   - Mathematical formulas with plain-language explanations
   - Common pitfalls and numerical stability issues
   - What to expect from experiments

3. **Create the HTML guide** — Use the /frontend-design skill to create a polished single-page explainer:
   - Warm cream paper aesthetic with Fraunces + Newsreader fonts
   - Interactive playgrounds where possible (sliders, demos)
   - Hand-tuned SVG diagrams for key concepts
   - Section numbering, drop caps, sidenotes, pull quotes
   - KaTeX for math rendering
   - Task cards at the end: concepts + thinking, NO code solutions
   - Save as `[topic]_guide.html` in the assignment directory

4. **Create the MD guide** — A portable markdown version with:
   - All the same content adapted for markdown
   - LaTeX math via `$...$` and `$$...$$`
   - ASCII diagrams where HTML had SVGs
   - Tables replacing interactive elements
   - Save as `[topic]_guide.md` in the assignment directory

## Rules

- **TEACH, don't solve.** The user wants to understand the concepts so they can solve the questions themselves.
- Every section should have concrete examples with real numbers.
- Each task in the assignment gets a "task card" with concepts needed and thinking required — but NEVER the actual code implementation.
- Include a table of contents with anchor links.
- Update the hero/meta to reflect what's covered.
- Match the editorial "scientific manuscript" aesthetic established in previous guides.
