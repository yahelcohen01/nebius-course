# Retrieval-Augmented Generation — A Study Guide

**Assignment 2 · Nebius · Due 28.4.26**
**Generation model:** Llama-3.3-70B-Instruct · **Judge:** DeepSeek-V3-0324 · **Embedder:** BAAI/bge-small-en-v1.5 · **Reranker (opt):** BAAI/bge-reranker-base

> A guided tour of the parts, the prompts, the metrics, and the experiments — built specifically for the FinanceBench assignment.

---

## Contents

1. [Why RAG exists](#1-why-rag-exists)
2. [The four-step recipe](#2-the-four-step-recipe)
3. [Chunking, in practice](#3-chunking-in-practice)
4. [Embeddings & vector space](#4-embeddings--vector-space)
5. [FAISS & the index](#5-faiss--the-index)
6. [Retrieval & top-k](#6-retrieval--top-k)
7. [Prompting the generator](#7-prompting-the-generator)
8. [Evaluating a RAG pipeline](#8-evaluating-a-rag-pipeline)
9. [Ragas wiring for Nebius](#9-ragas-wiring-for-nebius)
10. [The FinanceBench dataset](#10-the-financebench-dataset)
11. [Improvement experiments](#11-improvement-experiments)
12. [Rerankers (cross-encoders)](#12-rerankers-cross-encoders)
13. [Multi-scale chunking (bonus)](#13-multi-scale-chunking-bonus)
14. [Task cards](#14-task-cards)

---

## §1 · Why RAG exists

Large language models are frozen on the day they stopped training. RAG is how you teach them about yesterday, about your private documents, and — most importantly — about the things they *don't know they don't know*.

An LLM trained in 2024 will answer a question about Q3 2025 earnings with the same fluent confidence it answers anything else. That's the problem: the model is always **tempted to answer**, even when it has no evidence. Worse, its context window has a limit, and even when the limit is huge, answer quality degrades when the relevant fact is buried in the middle of a million tokens (the "lost in the middle" effect).

RAG fixes this by **injecting a few carefully selected, highly relevant passages** into the prompt at query time, and then instructing the model to answer only from those passages. The model still does the writing — but the *facts* come from the retrieval step.

> **The one-liner rule.** Fine-tune teaches the model to *write differently*. RAG teaches the model to *write about something new*. If your problem is "the model needs to know about this 10-K," you want RAG — not fine-tuning.

*Course analogy:* fine-tune = parrot who read many books in a subject. RAG = parrot with a really good search engine.

---

## §2 · The four-step recipe

Every RAG system is four boxes. Two happen offline. Two happen every time a user hits enter.

```
INDEXING (offline, once)
┌─────────┐   ┌─────────┐   ┌─────────────┐
│ 1·Chunk │ → │ 2·Embed │ → │ FAISS index │
└─────────┘   └─────────┘   └─────────────┘
                                   │  (persisted to disk)
                                   ▼
INFERENCE (online, per query)
┌───────┐   ┌────────────┐   ┌────────────┐
│ Query │ → │ 3·Retrieve │ → │ 4·Generate │ → answer
└───────┘   └────────────┘   └────────────┘
```

| Step | When it runs | What can break |
|------|--------------|----------------|
| **Chunk** | Once, offline | Too small → no context. Too large → diluted embeddings. |
| **Embed + store** | Once, offline | Wrong model; metadata lost in splitting; wrong similarity metric. |
| **Retrieve** | Every query | Right doc not in top-k; noisy chunks; wrong filter. |
| **Generate** | Every query | Model ignores context, hallucinates, doesn't cite. |

> **For Task 2:** the prompt asks you to split your write-up into *indexing* (offline), *retrieval* (per query), and *generation* (per query). Use this as your scaffolding.

---

## §3 · Chunking, in practice

A chunk is the smallest unit you can possibly retrieve. Too small and it's meaningless ("this shall not apply in cases described above"). Too large and a single 8-topic embedding dilutes all 8 topics into mush.

### The three levers

| Lever | What it does | Assignment value |
|-------|--------------|------------------|
| `chunk_size` | Characters per chunk | `1000` |
| `chunk_overlap` | Characters repeated between adjacent chunks | `150` |
| Split strategy | Where to break: paragraph → line → sentence → word | `RecursiveCharacterTextSplitter` |
| Metadata | `doc_name`, `page_number`, `company`, `doc_period` | **attach before splitting** |

**Why 150 overlap?** Overlap at 10–20% of `chunk_size` is the textbook range. For `chunk_size=1000`, 150 is 15% — deep enough to bridge most mid-sentence breaks without wasting space on duplicates.

### The page → chunks picture

```
PAGE (one PyPDFLoader Document, ~1500 chars)
┌────────────────────────────────────────────────────────────┐
│ raw page text + metadata {doc_name, page_number=7, ...}    │
└────────────────────────────────────────────────────────────┘
                             ↓ split (1000 / 150)
┌────────────────────────────────────────────┐
│ chunk A · chars 0 – 999 · metadata inherited│
└────────────────────────────────────────────┘
                     ┌──────────────────┐
                     │ OVERLAP 150 chars│ ← same text in both chunks
                     └──────────────────┘
                             ┌──────────────────────────────────┐
                             │ chunk B · chars 850 – 1500       │
                             └──────────────────────────────────┘
```

Chunks **inherit page metadata** from the Document. The overlap strip is repeated in both chunks so nothing gets severed at a boundary.

### Why metadata matters *before* splitting

If you attach `{doc_name, page_number}` to the Document before calling the splitter, every child chunk carries it. If you try after splitting you have to recompute. The assignment insists on `page_number` as a **0-indexed** field so it can be compared directly to the dataset's `evidence_page_num`.

> **Gotcha.** `PyPDFLoader` already attaches a default `page` field. You'll either rename it or add a new `page_number`. Pick one — inconsistency here will haunt Task 6.

### Chunk-size intuition (500-page filing)

| chunk_size | ≈ chunks | Verdict |
|-----------:|---------:|---------|
| 300        | ~4700    | Tight — precise matches but facts often split |
| 500        | ~2800    | Snug — narrow retrieval, brittle for multi-sentence evidence |
| **1000**   | **~1650** | **Assignment default — good balance for 10-K prose** |
| 1500       | ~1040    | Wide — one embedding covers multiple ideas |
| 2000       | ~760     | Diluted — noisy retrieval, large prompt |

---

## §4 · Embeddings & vector space

An embedding model is a function that maps a string into a point in a high-dimensional space, such that strings with similar meaning land near each other.

For this assignment we use `BAAI/bge-small-en-v1.5`. Three numbers to remember:

| Property | Value | Note |
|----------|-------|------|
| Dimensions | **384** | each chunk → 384 floats |
| Similarity | **cosine** | direction only, ignores magnitude |
| Typical score range | **[0.6, 1.0]** | BGE's contrastive training scales scores up; 0.5 is *not* "similar" |

### Why cosine?

$$\mathrm{cos}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

It's a dot product after length-normalising both vectors. So "apple pie" and "a long, long essay about apple pie" land close even though one has much larger raw magnitude. Dot product would favour the long one. L2 mixes geometry with magnitude. Cosine is the right default for text.

### Good vs bad vector space

```
GOOD                                  BAD
  ▲                                    ▲
  │  (king, queen, prince)             │    king · · · apple
  │       ●                            │
  │      / \                           │    banana · prince
  │     ●   ●                          │
  │                                    │
  │          (apple, fruit, banana)    │  (no meaningful clusters)
  │                ● ● ●               │
  └──────────────────────────────▶      └──────────────────────────▶
```

A good embedding clusters semantically similar strings. A bad one scatters them. Everything downstream builds on this.

### Query vs passage symmetry

The bge models are trained so a query embedding and a chunk embedding can be compared directly with cosine. Some models require a `"query: ..."` prefix for asymmetric tasks. For FinanceBench — question vs filing prose — the difference is small; use LangChain's defaults unless retrieval is clearly underperforming.

---

## §5 · FAISS & the index

A vector store is a database tuned for one operation: "given a query vector, give me the k closest stored vectors, fast." FAISS (Facebook AI Similarity Search) is the one Meta built; it's fast, local, and what LangChain wraps.

### Build time

You hand FAISS the chunks (text + metadata) and an embedding function. FAISS embeds everything, stores the 384-dim vectors in an index, and keeps the text/metadata alongside.

### Query time

1. Your query goes through the same embedding model.
2. FAISS runs approximate nearest-neighbour (ANN) search.
3. It returns `k` Documents, each with the full metadata you attached at build time.

> **Save it to disk.** `vectorstore.save_local(path)` persists the FAISS index + the docstore with metadata. Reloading with `FAISS.load_local` needs `allow_dangerous_deserialization=True`. Save once, reuse across Tasks 4–7 — re-embedding every restart burns time and API calls.

*Numeric scale for the assignment:* ~40 filings × ~200–300 pages × chunk_size=1000 ≈ 15–25k chunks. Entirely fine for a local FAISS index on a laptop.

---

## §6 · Retrieval & top-k

Retrieval is the step most RAG pipelines lose on. Get this wrong and it doesn't matter how good your LLM is — it's answering from bad inputs.

### The k knob — two opposing effects

|                   | small k (e.g., 1) | large k (e.g., 10) |
|-------------------|-------------------|--------------------|
| **Recall**        | low (easy to miss) | high (evidence almost always in there) |
| **Precision**     | high (little noise) | low (many irrelevant chunks) |
| **Prompt size**   | small · cheap · fast | large · expensive · "lost in the middle" risk |
| **Faithfulness**  | may lack context | may be distracted |

### page-hit@k — how the assignment measures retrieval

For each question, FinanceBench tells you the exact page the evidence is on (`evidence_page_num`, 0-indexed). The metric is binary, per-question:

$$\text{page-hit@}k = \begin{cases} 1 & \text{if any top-}k\text{ chunk has matching page\_number} \\ 0 & \text{otherwise} \end{cases}$$

Average across the dataset = single number. Multi-evidence questions count as a hit if *any* evidence page is retrieved.

> **Common bug.** 0-indexed vs 1-indexed. PyPDFLoader gives 0-indexed. FinanceBench's `evidence_page_num` is 0-indexed. If you compare 1-indexed to 0-indexed, every hit becomes a near-hit and your number looks mysteriously awful.

### page-hit@k mini-worked-example

Suppose the correct page for Q7 is page 42, and your retriever returns these chunks:

| Rank | doc_name | page_number |
|-----:|----------|------------:|
| 1 | AAPL_10K.pdf | 39 |
| 2 | AAPL_10K.pdf | 42 ← match |
| 3 | AAPL_10K.pdf | 40 |
| 4 | GOOG_10K.pdf | 17 |
| 5 | AAPL_10K.pdf | 41 |

- page-hit@1 = 0 (rank 1 is page 39)
- page-hit@3 = 1 (match at rank 2)
- page-hit@5 = 1

### Retrieval failure modes

| Failure | Symptom | First thing to try |
|---------|---------|-------------------|
| Right doc not retrieved | page-hit@k low for many questions | Raise k; better embedder; reranker |
| Wrong ranking order | Right page at rank 5 when k=3 | Reranker |
| Context split across chunks | Chunk has half the fact | Larger overlap; larger chunk_size |
| Noise drowns signal | High k but faithfulness falls | Smaller k; reranker |

---

## §7 · Prompting the generator

Retrieval gives you facts; prompting decides how they're used. The exact same chunks fed through two different prompts can give completely different answers.

### A well-formed RAG prompt contains

1. A **system message** setting ground rules: context-only, say-IDK, cite.
2. A **context block** — chunks clearly separated, each tagged with `doc_name` + `page_number`.
3. The **user question**, verbatim.
4. A **closing instruction** reinforcing the grounding rule.

> **Empty retrieval is a case.** If the retriever returned nothing, don't hand the model an empty `CONTEXT:` block. Explicitly tell it "No relevant context was retrieved" and instruct it to refuse.

### The three grounding commandments

| # | Rule | Why |
|---|------|-----|
| 1 | **Answer only from the context.** | Prevents the model from reaching into training data. |
| 2 | **Say "I don't know" when the context lacks the answer.** | Without this, models hedge-hallucinate rather than refuse. |
| 3 | **Cite the `doc_name` for each claim.** | Makes faithfulness checkable and the answer more useful. |

### Chunk formatting matters

Think of the context block as a mini-document the model reads. Separators, source headers, and numbered sources all help the model attribute the right claim to the right chunk.

```
CONTEXT:
[Source 1 | doc: AAPL_2023_10K.pdf | page: 42]
<chunk text>

[Source 2 | doc: AAPL_2023_10K.pdf | page: 43]
<chunk text>

---
QUESTION: <user question>

Answer using only the above sources. Cite the document for each fact.
If the answer is not present, say so explicitly.
```

> "**Garbage retrieval → garbage generation.** The best grounding prompt in the world cannot rescue a retriever that missed the page."

---

## §8 · Evaluating a RAG pipeline

A single end-to-end accuracy number tells you the pipeline is broken but not where. Component-level metrics let you point to retrieval vs. generation and tune the guilty piece.

### The three metrics Task 6 wants

```
  RETRIEVAL             GENERATION            END-TO-END
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│ page-hit@k   │     │ faithfulness │      │ correctness  │
│ (k=1, 3, 5)  │     │ (ragas, n=20)│      │ (LLM-judge)  │
└──────────────┘     └──────────────┘      └──────────────┘
 "Did we get       "Did the model use      "Is the final
  the right         the retrieved chunks    answer right?"
  pages?"           honestly?"
```

### 1 · Correctness — LLM as judge

Compare the RAG answer to the gold answer, using a *different* model as judge (DeepSeek-V3-0324). Binary `{correct / incorrect}` + one-sentence justification. Using a different model than the generator avoids "marking its own homework."

Average across the dataset = overall correctness. Brittle on numeric answers — the judge has to recognise "4.2 billion" and "$4,200,000,000" as the same, so the judge prompt matters.

### 2 · Faithfulness — Ragas

Narrower than correctness: *is every claim in the answer supported by the context that was actually retrieved?* A correct-sounding answer can still be **unfaithful** if the model pulled it from training data instead of the retrieved chunks.

Ragas decomposes the answer into atomic claims and checks each against the context:

$$\text{Faithfulness} = \frac{\# \text{supported claims}}{\# \text{claims in answer}}$$

Because this is many LLM calls per example, the assignment restricts it to the **first 20 questions** only.

### 3 · page-hit@k — retrieval in isolation

Covered in §6. Cheap: binary check, no LLM call, run at k ∈ {1, 3, 5}.

### Why three metrics, not one

| Correctness | Faithfulness | page-hit@k | Diagnosis |
|:-----------:|:------------:|:---------:|-----------|
| High | Low | High | Right answer, but from memory — "lucky hallucination." |
| Low | High | High | Dutifully quoting the *wrong* chunks; retrieval returns related-but-wrong. |
| Low | High | Low | Faithful but retrieval missed the page — fix retrieval. |
| Low | Low | Low | End-to-end failure, probably retrieval first. |

Each combination points to a different fix.

> **Expect disappointment.** FinanceBench is *hard*. Vector-RAG numbers in the literature hover around 19–50% correctness on the full set. Don't chase a benchmark you won't hit — chase *insights* about which component is failing.

---

## §9 · Ragas wiring for Nebius

Ragas is a library. Nebius is an OpenAI-compatible API. The assignment wants them glued together with a specific wrench: **the collections API + an AsyncOpenAI client + `llm_factory`**.

### The four pieces

| Piece | What it does |
|-------|--------------|
| `AsyncOpenAI` | OpenAI's async client. Point at `base_url="https://api.studio.nebius.ai/v1/"` → talks to Nebius. |
| `ragas.llms.llm_factory` | Wraps that client into a shape Ragas metrics accept. |
| `ragas.metrics.collections.Faithfulness` | The new-style Faithfulness metric. Use this, not the older class-based one. |
| `.score(...)` | Synchronous entry point. Assignment explicitly wants this, **not** `.ascore`. |

### Conceptual shape of the call

```
async_client = AsyncOpenAI(api_key=..., base_url="https://api.studio.nebius.ai/v1/")
ragas_llm    = llm_factory(model="deepseek-ai/DeepSeek-V3-0324", client=async_client)
faith        = Faithfulness(llm=ragas_llm)

score = faith.score(
    user_input = question,
    response   = rag_answer,
    retrieved_contexts = [chunk.page_content for chunk in retrieved_chunks],
)
```

> **Watch for Ragas version skew.** The `.metrics.collections` namespace is new. If your import fails, check the installed version and upgrade.

---

## §10 · The FinanceBench dataset

150 expert-written questions over real US public-company filings (10-Ks, 10-Qs, earnings releases). Designed to be genuinely hard — the benchmark's own paper reports ~19% vector-RAG accuracy on the full set.

### Fields you'll touch

| Column | Meaning | How you'll use it |
|--------|---------|-------------------|
| `financebench_id` | unique question id | sort key for "first 5 per type"; xlsx row key |
| `question` | the question | input to naive + RAG |
| `answer` | gold answer | ground truth for correctness |
| `question_type` | `metrics-generated` / `domain-relevant` / `novel-generated` | drop first; stratify Task 1/5 selection |
| `doc_name` | which filing | metadata; spot-check retrieval |
| `doc_link` | PDF URL | some are dead — replace with course repo link |
| `evidence → evidence_page_num` | page(s) with answer, **0-indexed** | ground truth for page-hit@k |
| `evidence → evidence_text` | exact annotator snippet | sanity-check chunking / retrieval |
| `dataset_subset_label` | subset label | **ignore** (per PDF) |

### Question types, in plain English

| Type | Description | RAG tends to... |
|------|-------------|-----------------|
| `metrics-generated` *(dropped)* | Templated ratio questions, large + repetitive | — |
| `domain-relevant` | Questions a finance analyst would plausibly ask | help (RAG retrieves the filing section) |
| `novel-generated` | LLM-authored, may combine facts unusually | vary — often needs multi-page synthesis |

---

## §11 · Improvement experiments

Task 7 asks for real experimental discipline: one hypothesis per experiment, change one thing from baseline, measure the same three metrics, compare.

### The five knobs you can turn

| Knob | Expected primary effect | Watch out for |
|------|-------------------------|---------------|
| **Generation prompt** | ↑ faithfulness | Over-strict prompts → refuses valid answers |
| **Generation model** | ↑ correctness & faithfulness (smarter reader) | Confounds size with training; don't over-interpret |
| **k in top-k** | ↑ page-hit@k; faithfulness may ↓ with distractors | Correctness — bigger prompts aren't always better |
| **chunk_size** | Shifts page-hit@k and correctness | Requires rebuilding FAISS — save under new names |
| **Reranker** | ↑ page-hit@k (correct page moves into top-4) | Adds latency |

> **One variable at a time.** If you change both the prompt *and* k and correctness improves, you won't know which one mattered. Baseline → change 1 → measure → back to baseline → change 2 → measure.

### What makes a good hypothesis

A good hypothesis names the metric you expect to move and the reason.

*Example:* "I expect the reranker to raise page-hit@4 by ~0.1 points because the correct page is often in the top-20 but ranked below rank 4 due to BGE's bi-encoder approximation. Correctness should track upward but more weakly — retrieval isn't the only failure mode."

That's testable. It names the metric (page-hit@4), the expected direction (up ~0.1), and the causal story (bi-encoder mis-ranks within top-20).

> **Faithfulness subsample.** The assignment fixes faithfulness on the first 20 questions across all experiments. Do the same 20 every time — otherwise you're comparing apples to differently-ripe apples.

---

## §12 · Rerankers (cross-encoders)

A **bi-encoder** (the embedding model) compresses query and chunk into vectors and compares them. A **cross-encoder** reads both together and scores them directly. The cross-encoder is much more accurate but too slow to run on every chunk — so you use both.

### Two-stage retrieval

```
STAGE 1 · BI-ENCODER           STAGE 2 · CROSS-ENCODER
┌────────────────────┐          ┌────────────────────────┐
│ FAISS              │          │ BAAI/bge-reranker-base │
│ top-20             │   →      │ top-4                  │   →   generator
│ fast, approximate  │          │ slow, joint, accurate  │
└────────────────────┘          └────────────────────────┘
```

- The bi-encoder embeds query and chunk separately → compare to all chunks in <1 s.
- The cross-encoder reads (query, chunk) pairs jointly → precise score but O(top-20) calls.
- Two-stage retrieval = cheap recall first, accurate precision second.

### Why keep k the same at the generator?

The assignment is emphatic: when varying k, keep *the k reaching the generator* as your variable; don't also change retrieve-width. With a reranker: retrieve top-20 → rerank → keep top-4 for the generator. The comparison is "4 chunks chosen by bi-encoder alone" vs "4 chunks chosen by bi-encoder-then-cross-encoder."

> **Cost.** bge-reranker-base runs on CPU but adds ~100 ms per query. For a ~130-question eval, manageable.

---

## §13 · Multi-scale chunking (bonus)

The hypothesis from the referenced article: no single chunk size is optimal for all queries. A question asking for a single number probably wants tight chunks; a question asking about a 3-paragraph strategic narrative wants wide chunks.

### What you're measuring

Build 2–3 FAISS indices with different `chunk_size` values (e.g., 300, 1000, 2000) and keep everything else identical. For each question, compute page-hit@5 from each index. Then ask:

1. **Per-question winner.** For how many questions does the "best" chunk size differ? If it's 10% or less, one winner dominates. If it's 40%+, chunk size is really query-dependent.
2. **Aggregate winner.** Average page-hit@5 per index. Is one chunk size clearly best overall?

### Three possible outcomes

- **(a)** One chunk size wins almost always → article's claim is weak here.
- **(b)** Different sizes win on different questions, no clear aggregate winner → article's claim is strong; multi-scale retrieval would help.
- **(c)** One size wins on average but the disagreement rate is high → article's claim partially holds; a hybrid strategy is worth exploring.

---

## §14 · Task cards

Per-task concept checklists. **No code solutions** — just what to think about, in what order, using which section of this guide.

---

### Task 1 — Naive generation · 10 pts

Ask Llama-3.3-70B 10 questions directly (5 domain-relevant + 5 novel-generated, first 5 per type sorted by `financebench_id`). No retrieval.

**Thinking:**
- Filter the dataset: drop `metrics-generated`.
- Sort by `financebench_id`, take first 5 per remaining type.
- Call the chat-completions endpoint once per question — nothing else in the prompt.
- Record verdict manually: `{correct, partially correct, wrong, refused}`. Refusals are their own category, not "wrong."
- Discussion: patterns by type, confident-but-wrong (hallucinations), refusals — *why*?

**Concepts:** §1 Why RAG (baseline motivation) · §10 FinanceBench fields.

---

### Task 2 — RAG reminder · 5 pts

Write ~2–4 sentences per component (indexing / retrieval / generation) answering: contribution? failure modes? offline or per-query?

**Thinking:**
- Cover offline vs online (§2).
- Name concrete failure modes (§3 chunking, §6 retrieval, §7 generation).
- Bonus: note the coupling — retrieval failure *causes* generation failure.

**Concepts:** §2 · §3 · §6 · §7.

---

### Task 3 — Embed documents & build FAISS · 15 pts

Load filtered PDFs, attach metadata, chunk (1000/150), embed with bge-small, store FAISS, spot-check retrieval.

**Thinking:**
- Load only PDFs whose `doc_name` is in the filtered dataset.
- **Attach metadata before splitting** — `doc_name`, `company`, `doc_period`, `page_number` (0-indexed).
- `RecursiveCharacterTextSplitter` with `chunk_size=1000, chunk_overlap=150`.
- Embed with `BAAI/bge-small-en-v1.5` from HuggingFace.
- Save FAISS to disk — reuse for Tasks 4–7.
- Spot-check: 2–3 questions; verify right doc, right page, chunks near evidence text.

**Concepts:** §3 chunking · §4 embeddings · §5 FAISS · §6 retrieval spot-check.

---

### Task 4 — Build the RAG pipeline · 25 pts

One function: `answer_with_rag(query, k=4) -> dict` returning `{answer, retrieved_chunks}`.

**Thinking:**
- System prompt with the three commandments: context-only, say-IDK, cite-source.
- Context block with clear separators + `doc_name` per chunk (§7).
- Handle empty retrieval explicitly (don't hand over an empty CONTEXT).
- Return raw chunks alongside the answer — later tasks need them.

**Concepts:** §7 prompt construction · §6 retrieval.

---

### Task 5 — Run & compare (naive vs RAG) · 10 pts

Same 10 questions from Task 1 through the new pipeline. Side-by-side discussion.

**Thinking:**
- Reuse Task 1 answers — don't recompute; apples-to-apples.
- For "RAG hurt" cases: diagnose why — wrong filing retrieved, or right filing + confusing chunks?
- Patterns by type: hypothesis on why domain-relevant might benefit differently from novel-generated.

**Concepts:** §1 RAG's theoretical advantages · §6 retrieval failures.

---

### Task 6 — Evaluation — 3 metrics, full dataset · 20 pts

Run every filtered question through RAG; score correctness + faithfulness (first 20 only) + page-hit@k for k ∈ {1, 3, 5}.

**Thinking:**
- Judge prompt for correctness: short, binary, one-sentence justification.
- Ragas Faithfulness via collections API, `.score()` (sync), first 20 sorted by `financebench_id`.
- page-hit@k: compare retrieved chunks' `page_number` to `evidence_page_num`; multi-evidence = hit if any matches.
- Report per-question xlsx + aggregate numbers.

**Concepts:** §8 the three metrics · §9 Ragas wiring.

---

### Task 7 — Improvement cycles · 15 pts

At least three one-variable-at-a-time experiments with hypothesis → change → measure → interpret.

**Thinking:**
- Freeze the faithfulness subsample at the same 20 questions from Task 6.
- Results table: baseline as first row, then each experiment.
- Write the hypothesis *before* seeing the numbers — if you're always right, your hypotheses are too vague.
- Wrap-up paragraph: where does the pipeline fail most? What would you try next?

**Concepts:** §11 five knobs · §12 rerankers.

---

### Bonus — Multi-scale chunking · +10 pts

Build 2–3 FAISS indices at different chunk sizes. Compare per-question page-hit@5.

**Thinking:**
- Keep embedding model, splitter type, overlap policy fixed — *only* chunk size changes.
- Save each index under a distinct name.
- Summary table + disagreement rate + discussion of whether the article's claim holds.

**Concepts:** §3 chunking · §13 multi-scale.

---

*Build the baseline first. Then change one thing at a time. A pipeline whose weakness you **understand** is already on its way to being better.*
