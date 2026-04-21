---
name: read-hometask
description: Read a notebook or assignment file, extract all tasks, questions, and point values, and produce a structured summary. Use when starting a new assignment or when the user says "what do I need to do", "read the hometask", "extract the tasks".
argument-hint: [file-path]
user-invocable: true
---

# Read Hometask

Read and analyze the assignment file: **$ARGUMENTS**

## Process

1. **Read the file** — Open the notebook or assignment file. For `.ipynb` files, search for task headers, point values, and TODO markers.

2. **Extract all tasks** — For each task, capture:
   - Task number and title
   - Point value
   - What needs to be implemented (code)
   - What needs to be answered (conceptual questions)
   - What needs to be visualized (plots, heatmaps)
   - What needs to be written (analysis paragraphs)
   - Any specific parameters or configurations mentioned

3. **Identify dependencies** — Note which tasks build on earlier ones (e.g., "use your function from Task 1.2").

4. **Present as a structured summary** — Output a clean table:

| Task | Title | Points | Type | Dependencies |
|------|-------|--------|------|-------------|

Then for each task, a bullet list of specific deliverables.

5. **Suggest a reading order** — Map tasks to the study guide sections if one exists.

## Rules

- Be exhaustive — don't miss any sub-task or hidden requirement.
- Quote exact parameter values mentioned in the assignment (learning rates, batch sizes, lambda values).
- Note the skeleton code provided vs what needs to be filled in.
- Flag any ambiguous requirements.
