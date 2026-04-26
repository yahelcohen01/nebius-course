# Retrieval-Augmented Generation — A Study Guide

**Assignment 2 · Nebius · Due 28.4.26**
**Generation model:** Llama-3.3-70B-Instruct · **Judge:** DeepSeek-V3-0324 · **Embedder:** BAAI/bge-small-en-v1.5 · **Reranker (opt):** BAAI/bge-reranker-base

> A guided tour of the parts, the prompts, the metrics, and the experiments — built specifically for the FinanceBench assignment.

---

## Contents

1. [Why RAG exists](#1-why-rag-exists)
2. [The FinanceBench dataset](#2-the-financebench-dataset)
3. [The four-step recipe](#3-the-four-step-recipe)
4. [Chunking, in practice](#4-chunking-in-practice)
5. [Embeddings & vector space](#5-embeddings--vector-space)
6. [FAISS & the index](#6-faiss--the-index)
7. [Retrieval & top-k](#7-retrieval--top-k)
8. [Prompting the generator](#8-prompting-the-generator)
9. [Evaluating a RAG pipeline](#9-evaluating-a-rag-pipeline)
10. [Ragas wiring for Nebius](#10-ragas-wiring-for-nebius)
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

## §2 · The FinanceBench dataset

Before you build anything: understand the thing you're being evaluated on. **FinanceBench** (Islam, Kannappan, Kiela, Qian, Scherrer & Vidgen, Patronus AI / Contextual AI / Stanford, 2023) was released as a first-of-its-kind test suite for *open-book* financial QA. It's deliberately hard — the benchmark's own paper showed GPT-4-Turbo + shared vector store **incorrectly answered or refused 81% of questions**.

### At a glance

| | |
|---|---|
| **10,231** | question–answer–evidence triplets |
| **40** | US public companies |
| **360** | public filings, 2015–2023 |
| **150** | questions in the open-source eval sample |
| **9 of 11** | GICS sectors represented |
| **20** | domain-expert annotators (finance background) |

### What's in each filing

The knowledge base is dense: **10-Ks** (270 of 360 documents, ~75%, and **93% of all questions**), plus 10-Qs (27), 8-Ks (29), earnings reports (29), and annual reports (5). A single 10-K runs 100–300 pages of mostly narrative prose with some tables — exactly the regime where plain keyword search fails and semantic retrieval has to earn its keep.

> **Why so many 10-Ks.** 10-Ks are comprehensive annual reports containing the most technical detail — income statement, balance sheet, cash flow, risk factors, management discussion. They're the canonical document financial analysts reach for.

### The three question types

Every question is labelled with one of three types. Each came from a different process and has very different properties. The assignment **drops `metrics-generated`** and keeps the other two.

| Type | Count | How generated | Difficulty |
|------|------:|---------------|------------|
| **domain-relevant** | 925 | 25 generic analyst questions asked against 37 companies ("did they pay a dividend in the last year?", "are operating margins consistent?"). Developed with financial analysts and refined by reviewing filings. | Medium. Some could be answered from general world knowledge. |
| **novel-generated** | 1,323 | Annotator-written questions specific to the company + report + industry. Brief: "realistic, varied, challenging" — not purely extractive, must involve reasoning. | Variable. May require synthesis across passages. |
| **metrics-generated** *(dropped)* | 7,983 | Templated from 18 base metrics × 8 years (2015–2022) × 32 companies, then augmented with derivative metrics like net-income margin. 11 "vanilla" + 11 "creative" intros and 7+7 endings, for phrasing diversity. | Mostly numerical reasoning. Often needs multiple financial statements. |

### Reasoning taxonomy

The paper also labelled *what kind of reasoning* each question requires:

| Type | Share |
|------|------:|
| Information extraction | 28% |
| Numerical reasoning | 66% |
| Logical reasoning | 6% |

The vast majority (66%) require the model to *do maths on numbers it just retrieved* — a known LLM weakness. Information-extraction questions, by contrast, are mostly "find the sentence, copy it out" and should behave best under RAG. Use this when discussing Task 5 ("why RAG helps some question types more than others").

### Why it's hard — the paper's own numbers

The FinanceBench paper tested 16 model configurations on the 150-question eval sample. Headline results establish the difficulty ceiling:

| Model configuration | Correct | Incorrect | Failed to answer |
|---------------------|--------:|----------:|-----------------:|
| GPT-4-Turbo · Closed Book | **9%** | 3% | 88% |
| GPT-4-Turbo · Shared Vector Store | **19%** | 13% | 68% |
| Llama 2 · Shared Vector Store | **19%** | **70%** | 11% |
| Llama 2 · Single Vector Store | **41%** | 54% | 5% |
| GPT-4-Turbo · Single Vector Store | **50%** | 11% | 39% |
| Claude 2 · Long Context | **76%** | 21% | 3% |
| GPT-4-Turbo · Long Context | **79%** | 17% | 4% |
| GPT-4-Turbo · Oracle* | **85%** | 15% | 0% |

\* Oracle = the model is handed the evidence page directly (no retrieval).

Three things to take away:

- **Retrieval is the bottleneck.** The gap between "shared vector store" (19%) and "single store per document" (50%) is 31 points — the *only* thing that changed is how narrowly the retriever had to search. Better retrieval → much better answers.
- **Hallucination, not refusal, is the main failure.** Llama 2 with a shared store answered wrong on 70% of questions but refused only 11%. Models would rather confidently lie than admit ignorance.
- **Oracle is ~85%, not 100%.** Even when the model is handed the exact evidence page, ~15% of questions are still wrong — numerical reasoning and multi-statement synthesis are genuine generation failures.

> **What this means for your targets.** If your end-to-end correctness lands near 20–50% on the full filtered set, you're in the range the paper reported for vector-RAG — not a disaster, it's the benchmark's difficulty. The assignment isn't graded on absolute score; it's graded on whether you can explain *which component* is failing.

### Qualitative failure modes (from the paper)

The paper's authors manually reviewed every response. They identified five recurring themes — bring these to Task 1 and Task 5 discussion:

- **Hallucinations.** Superficially coherent, fully-justified, with reasoning steps — but wrong. Most dangerous because they're hardest to catch.
- **Helpful refusals.** Model refuses but tells the user where to find the info. Useful, but still a failure by the metric.
- **Different but valid.** Model answers differently from the gold label and is still correct — especially for qualitative questions. The paper's authors allowed these as correct.
- **Irrelevant comments.** Model doesn't understand the task and "guesses" in a generic direction.
- **High-quality correct.** Better than the gold answer — with multiple evidence points, absolute + percentage diffs, full calculation steps.
- **Unit / rounding confusion.** Correct logic, wrong units (millions vs billions), or tiny rounding error. Paper's judges allowed *minor* deviations.

### Dataset fields you'll touch

| Column | Meaning | How you'll use it |
|--------|---------|-------------------|
| `financebench_id` | unique id (`financebench_id_0000` style) | sort key for "first 5 per type"; xlsx row key |
| `question` | the question text | input to naive + RAG |
| `answer` | human-annotated gold answer | ground truth for correctness |
| `question_type` | `metrics-generated` / `domain-relevant` / `novel-generated` | drop `metrics-generated`; stratify Task 1/5 selection |
| `justification` | annotator's reasoning (where relevant) | sanity-check a "wrong" RAG answer that might be right |
| `doc_name` | the specific filing | metadata on every chunk; spot-check retrieval |
| `company`, `gics_sector`, `doc_period`, `doc_type` | filing context | optional metadata filtering |
| `doc_link` | PDF URL | some are dead — replace with course repo link |
| `evidence → evidence_page_num` | page(s) containing the answer | ground truth for page-hit@k |
| `evidence → evidence_text` | exact annotator-chosen snippet | sanity-check chunking + retrieval |
| `evidence → evidence_text_full_page` | entire page around the evidence text | extra context for the spot-check |
| `dataset_subset_label` | subset flag | **ignore** (per the assignment PDF) |

> **The 0-indexed vs 1-indexed trap.** The FinanceBench paper notes "all page numbers are 1-indexed" in the raw GitHub JSONL. But the HuggingFace `PatronusAI/financebench` dataset lists `evidence_page_num` as 0-indexed, and the assignment explicitly says "keep `page_number` 0-indexed to match `evidence_page_num`." Bottom line: after loading the dataset, **print one row and one PyPDFLoader page to verify they align.** Mis-alignment by one silently wrecks page-hit@k.

### What the assignment keeps vs drops

After dropping `metrics-generated`, you're left with **~2,248 questions** (925 + 1,323) in the full dataset. The open-source eval sample splits 50/50/50 across types, so when you filter it, you'll have roughly **100 questions** (50 domain-relevant + 50 novel-generated) to run — a practical size for Tasks 1, 5, 6, 7.

> **Companies you'll likely see.** The paper's open-source sample draws heavily from 3M, Boeing, CVS Health, Coca-Cola, MGM Resorts, Netflix, Pfizer, Salesforce, Ulta Beauty, and Verizon — but the filtered set can include others. The "more documents in the folder than referenced in the dataset" note in the assignment means: **don't index the whole folder, only the filings actually referenced after filtering.**

---

## §3 · The four-step recipe

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

## §4 · Chunking, in practice

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

If you attach `{doc_name, page_number}` to the Document before calling the splitter, every child chunk carries it. If you try after splitting you have to recompute. The assignment insists on `page_number` as a **0-indexed** field so it can be compared directly to the dataset's `evidence_page_num` (see the 1-vs-0-indexed warning in §2).

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

## §5 · Embeddings & vector space

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

## §6 · FAISS & the index

A vector store is a database tuned for one operation: "given a query vector, give me the k closest stored vectors, fast." FAISS (Facebook AI Similarity Search) is the one Meta built; it's fast, local, and what LangChain wraps.

### Build time

You hand FAISS the chunks (text + metadata) and an embedding function. FAISS embeds everything, stores the 384-dim vectors in an index, and keeps the text/metadata alongside.

### Query time

1. Your query goes through the same embedding model.
2. FAISS runs approximate nearest-neighbour (ANN) search.
3. It returns `k` Documents, each with the full metadata you attached at build time.

> **Save it to disk.** `vectorstore.save_local(path)` persists the FAISS index + the docstore with metadata. Reloading with `FAISS.load_local` needs `allow_dangerous_deserialization=True`. Save once, reuse across Tasks 4–7 — re-embedding every restart burns time and API calls.

*Numeric scale for the assignment:* the filtered FinanceBench subset (~40 filings × ~200–300 pages) × chunk_size=1000 ≈ 15–25k chunks. Entirely fine for a local FAISS index on a laptop.

---

## §7 · Retrieval & top-k

Retrieval is the step most RAG pipelines lose on. Get this wrong and it doesn't matter how good your LLM is — it's answering from bad inputs. The FinanceBench paper showed this directly: going from shared-store to single-store retrieval alone moved GPT-4-Turbo from 19% to 50% correct (see §2).

### The k knob — two opposing effects

|                   | small k (e.g., 1) | large k (e.g., 10) |
|-------------------|-------------------|--------------------|
| **Recall**        | low (easy to miss) | high (evidence almost always in there) |
| **Precision**     | high (little noise) | low (many irrelevant chunks) |
| **Prompt size**   | small · cheap · fast | large · expensive · "lost in the middle" risk |
| **Faithfulness**  | may lack context | may be distracted |

### page-hit@k — how the assignment measures retrieval

For each question, FinanceBench tells you the exact page the evidence is on (`evidence_page_num`). The metric is binary, per-question:

$$\text{page-hit@}k = \begin{cases} 1 & \text{if any top-}k\text{ chunk has matching page\_number} \\ 0 & \text{otherwise} \end{cases}$$

Average across the dataset = single number. Multi-evidence questions count as a hit if *any* evidence page is retrieved.

> **Common bug.** 0-indexed vs 1-indexed page numbers. See the warning in §2 — verify by printing one dataset row side by side with a PyPDFLoader page before trusting any page-hit number.

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

## §8 · Prompting the generator

Retrieval gives you facts; prompting decides how they're used. The exact same chunks fed through two different prompts can give completely different answers. The FinanceBench paper found a 50-point swing from prompt-order alone (Context-First: 78%, Context-Last: 25%).

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

## §9 · Evaluating a RAG pipeline

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

Average across the dataset = overall correctness. Brittle on numeric answers — the judge has to recognise "4.2 billion" and "$4,200,000,000" as the same, so the judge prompt matters. The FinanceBench paper's human judges explicitly allowed small rounding and unit conversions as correct; your judge prompt should too (see §2 qualitative failure modes).

### 2 · Faithfulness — Ragas

Narrower than correctness: *is every claim in the answer supported by the context that was actually retrieved?* A correct-sounding answer can still be **unfaithful** if the model pulled it from training data instead of the retrieved chunks.

Ragas decomposes the answer into atomic claims and checks each against the context:

$$\text{Faithfulness} = \frac{\# \text{supported claims}}{\# \text{claims in answer}}$$

Because this is many LLM calls per example, the assignment restricts it to the **first 20 questions** only.

### 3 · page-hit@k — retrieval in isolation

Covered in §7. Cheap: binary check, no LLM call, run at k ∈ {1, 3, 5}.

### Why three metrics, not one

| Correctness | Faithfulness | page-hit@k | Diagnosis |
|:-----------:|:------------:|:---------:|-----------|
| High | Low | High | Right answer, but from memory — "lucky hallucination." |
| Low | High | High | Dutifully quoting the *wrong* chunks; retrieval returns related-but-wrong. |
| Low | High | Low | Faithful but retrieval missed the page — fix retrieval. |
| Low | Low | Low | End-to-end failure, probably retrieval first. |

Each combination points to a different fix.

> **Expect disappointment.** See §2 for the paper's own numbers — vector-RAG clusters around 19–50% correctness. Don't chase a benchmark you won't hit. Chase *insights* about which component is failing.

---

## §10 · Ragas wiring for Nebius

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

> **Cost.** bge-reranker-base runs on CPU but adds ~100 ms per query. For a ~100-question eval, manageable.

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
- Discussion: patterns by type, confident-but-wrong (hallucinations), refusals — *why*? Use the failure-mode vocabulary from §2.

**Concepts:** §1 Why RAG (baseline motivation) · §2 FinanceBench types + failure modes.

---

### Task 2 — RAG reminder · 5 pts

Write ~2–4 sentences per component (indexing / retrieval / generation) answering: contribution? failure modes? offline or per-query?

**Thinking:**
- Cover offline vs online (§3).
- Name concrete failure modes (§4 chunking, §7 retrieval, §8 generation).
- Bonus: note the coupling — retrieval failure *causes* generation failure.

**Concepts:** §3 · §4 · §7 · §8.

---

### Task 3 — Embed documents & build FAISS · 15 pts

Load filtered PDFs, attach metadata, chunk (1000/150), embed with bge-small, store FAISS, spot-check retrieval.

**Thinking:**
- Load only PDFs whose `doc_name` is in the filtered dataset. (Don't index the whole folder — it has more PDFs than you need.)
- **Attach metadata before splitting** — `doc_name`, `company`, `doc_period`, `page_number` (0-indexed; verify against the dataset — see §2).
- `RecursiveCharacterTextSplitter` with `chunk_size=1000, chunk_overlap=150`.
- Embed with `BAAI/bge-small-en-v1.5` from HuggingFace.
- Save FAISS to disk — reuse for Tasks 4–7.
- Spot-check: 2–3 questions; verify right doc, right page, chunks near evidence text.

**Concepts:** §4 chunking · §5 embeddings · §6 FAISS · §7 retrieval spot-check.

---

### Task 4 — Build the RAG pipeline · 25 pts

One function: `answer_with_rag(query, k=4) -> dict` returning `{answer, retrieved_chunks}`.

**Thinking:**
- System prompt with the three commandments: context-only, say-IDK, cite-source.
- Context block with clear separators + `doc_name` per chunk (§8).
- Handle empty retrieval explicitly (don't hand over an empty CONTEXT).
- Return raw chunks alongside the answer — later tasks need them.

**Concepts:** §8 prompt construction · §7 retrieval.

---

### Task 5 — Run & compare (naive vs RAG) · 10 pts

Same 10 questions from Task 1 through the new pipeline. Side-by-side discussion.

**Thinking:**
- Reuse Task 1 answers — don't recompute; apples-to-apples.
- For "RAG hurt" cases: diagnose why — wrong filing retrieved, or right filing + confusing chunks?
- Patterns by type: hypothesis on why domain-relevant might benefit differently from novel-generated. Lean on §2's reasoning taxonomy — extraction-heavy questions should benefit most from RAG.

**Concepts:** §1 RAG's theoretical advantages · §2 question types · §7 retrieval failure modes.

---

### Task 6 — Evaluation — 3 metrics, full dataset · 20 pts

Run every filtered question through RAG; score correctness + faithfulness (first 20 only) + page-hit@k for k ∈ {1, 3, 5}.

**Thinking:**
- Judge prompt for correctness: short, binary, one-sentence justification. Allow minor unit / rounding drift (§2 paper practice).
- Ragas Faithfulness via collections API, `.score()` (sync), first 20 sorted by `financebench_id`.
- page-hit@k: compare retrieved chunks' `page_number` to `evidence_page_num`; multi-evidence = hit if any matches.
- Report per-question xlsx + aggregate numbers.

**Concepts:** §9 the three metrics · §10 Ragas wiring.

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

**Concepts:** §4 chunking · §13 multi-scale.

---

*Build the baseline first. Then change one thing at a time. A pipeline whose weakness you **understand** is already on its way to being better.*
