# A Field Guide to QLoRA Fine-Tuning

*For LLM Architectures Hometask 3, Part 1 (75 points)*

On bending a frozen, four-bit language model toward a new task by training thin matrices on its edges — and why this trick lets a 360-million-parameter model learn a new behavior on your laptop.

**Tools:** transformers · peft · bitsandbytes · trl
**Read time:** ~45 minutes

---

## Contents

1. [Why Fine-Tune at All](#1-why-fine-tune-at-all)
2. [The Trinity: Model + Adapter + Quantization](#2-the-trinity-model--adapter--quantization)
3. [What 4-Bit Quantization Actually Means](#3-what-4-bit-quantization-actually-means)
4. [LoRA: The Low-Rank Trick](#4-lora-the-low-rank-trick)
5. [Causal vs Seq2Seq — Two Architectures, Two Formats](#5-causal-vs-seq2seq--two-architectures-two-formats)
6. [Building the Dataset](#6-building-the-dataset-task-11)
7. [The Training Recipe](#7-the-training-recipe)
8. [Evaluation — Reading Two Generations Side by Side](#8-evaluation--reading-two-generations-side-by-side)
9. [Common Failure Modes](#9-common-failure-modes)
10. [Task Cards](#10-task-cards)

---

## 1. Why Fine-Tune at All

> *A pre-trained model is a tourist with a phrasebook. Fine-tuning is the year it spends living in the country.*

Pre-trained language models are general by construction. They have read an enormous slice of the internet and learned the statistical shape of language — but they have not been asked to do *your* particular thing. SmolLM2 will happily answer questions; it will not, without coaching, modulate its tone between "child" and "expert" reliably.

Three ways to get a model to do what you want:

| Method | How it works | When it fits |
|---|---|---|
| **Prompt engineering** | Write a clever instruction and a few examples in the context window. | Free, instant, no training. Sufficient when the model already knows how to do the task and just needs to be told. |
| **Full fine-tuning** | Update *every* parameter on a new dataset. | Maximum control. Requires GPU memory proportional to the full model (gradient, optimizer state, weights). For a 360M model that already means tens of GBs. |
| **LoRA / QLoRA** | Freeze the base model. Train a small bolt-on adapter (LoRA), optionally with the base loaded in 4 bits (QLoRA). | You want a real behavior change but cannot afford full fine-tuning. The default modern recipe for instruction tuning on a single GPU. |

The hometask asks you to push the model's behavior past what a prompt alone reliably achieves: three distinct registers (child / student / expert) for the *same* question. That is exactly the regime where fine-tuning earns its keep.

> **A LoRA adapter is the model's accent, learned in an afternoon.**

---

## 2. The Trinity: Model + Adapter + Quantization

> *Three moving parts. Get all three right and a 360M model trains on a free-tier GPU.*

QLoRA is a stack of three independent ideas. None of them is the whole trick; together they unlock single-GPU fine-tuning of large models.

```
┌────────────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  FROZEN BASE MODEL     │     │  LORA ADAPTER    │     │   ADAPTED MODEL      │
│                        │     │                  │     │                      │
│   large W (4-bit NF4)  │  +  │      A · B       │  =  │   W' = W + α·A·B     │
│   millions of params   │     │   rank r (e.g.8) │     │                      │
│   no gradient   ❄      │     │   trainable 🔥   │     │   acts fine-tuned    │
└────────────────────────┘     └──────────────────┘     └──────────────────────┘
   stored once, never           ~0.1–1% of                on-the-fly during
   updated                      base params               forward pass
```

1. **Quantization** shrinks the base model's memory footprint by storing weights in 4 bits instead of 16 or 32. The model becomes cheap to *hold*.
2. **LoRA** injects tiny trainable matrices alongside the frozen weights. Only those tiny matrices receive gradients. The model becomes cheap to *update*.
3. **The training stack** (PEFT + bitsandbytes + Trainer) glues them together: dequantize on the fly, run the LoRA path in 16-bit, backprop only through the adapter.

---

## 3. What 4-Bit Quantization Actually Means

> *Sixteen bins, chosen so weights land in the right places. Two saved passes through the bin table. That is the whole idea.*

A neural network weight is conventionally a 32-bit float — four bytes of dynamic range per number. Four-bit quantization replaces it with one of 16 possible values. The naive way (uniform spacing from −1 to +1) wastes resolution: most weights sit near zero, and the extremes are mostly empty. QLoRA uses a smarter encoding called **NF4** (Normal Float 4).

### NF4: bins chosen by quantile

Neural-network weights are roughly Gaussian, centered at zero. NF4 picks its 16 codepoints by *quantile*: each of the 16 bins represents an equal slice of the normal distribution. The bins cluster densely near zero where the weights are crowded, and spread out at the tails.

```
weight density (Gaussian):
                  ▁▂▃▅▇█▇▅▃▂▁
NF4 bins:        ││││├┤├┤├┤││││││
                  (dense in center, sparse at tails)

uniform bins:    │  │  │  │  │  │  │  │
                  (wastes codes on empty tails)
```

*Why information-theoretically optimal?* If each codepoint is equally likely a priori, you use all 16 of them. Uniform spacing wastes most codes on empty tails.

### Double quantization

Quantization stores not just bin indices but also a scaling constant per block of weights (telling the decoder how big the bins are). Those constants are themselves floats. **Double quantization** quantizes them too — saving ~0.5 bit per parameter. It is bookkeeping, but at billions of parameters bookkeeping matters.

### The trick that makes it work

The 4-bit weights are *stored* at low precision but *computed* in 16-bit. During every forward pass, the relevant weight block is dequantized to bfloat16, the matrix multiply runs in 16-bit, and the result continues. You pay a small compute cost to keep numerical stability.

```python
# BitsAndBytesConfig — the four lines that do all of this
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                              # store weights as 4-bit
    bnb_4bit_quant_type="nf4",                       # use NF4 (vs "fp4")
    bnb_4bit_use_double_quant=True,                  # quantize the constants too
    bnb_4bit_compute_dtype=torch.bfloat16,           # compute in bf16
)
```

> **What you actually save:** for SmolLM2-360M, full-precision weights need roughly 1.4 GB. In 4-bit NF4 + double quantization, the same model fits in ~0.25 GB. That is the whole difference between "won't load on a free Colab" and "runs comfortably."

---

## 4. LoRA: The Low-Rank Trick

> *If the update you need is low-rank, you do not need to store it at full rank. Two thin matrices will do.*

Fine-tuning a transformer means modifying its weight matrices — typically the `q`, `k`, `v`, `o` projections inside attention, and possibly the MLP layers. A naive update is a delta matrix the same size as the original:

$$W' = W + \Delta W$$

If $W$ is $1024 \times 1024$ then so is $\Delta W$ — over a million numbers to store and update per layer. The LoRA observation is that for most task adaptations, *$\Delta W$ does not need to be full-rank*. The actual change in behavior lives in a low-dimensional subspace. So we parameterize:

$$\Delta W = \frac{\alpha}{r} \cdot A B \quad \text{where} \quad A \in \mathbb{R}^{d \times r}, \ B \in \mathbb{R}^{r \times d}$$

With $r$ typically 8 or 16. The full adapted matrix is then:

$$W'x = Wx + \frac{\alpha}{r} A B x$$

```
        W                A      B                  W'
   ┌─────────┐        ┌─┐   ┌─────────┐        ┌─────────┐
   │         │        │ │   │  r × d  │        │         │
   │  d × d  │   +    │ │ · └─────────┘   =    │ adapted │
   │ frozen  │        │ │                      │         │
   │         │        │ │                      │         │
   └─────────┘        └─┘                      └─────────┘
   ~1 048 576       trainable rank-r            forward
   params           ~16 384 params              pass uses W'
```

### The three knobs: r, alpha, target_modules

| Knob | What it controls | How to think about it |
|---|---|---|
| `r` | The rank — the resolution of the adaptation. | Low `r` = blurry sketch of the new behavior. High `r` = sharper but more parameters and overfitting risk. Common values: 4, 8, 16, 32. For a small task on a small model, `r=8` is plenty. |
| `lora_alpha` | Scaling factor. Effective update is `(α / r) · AB`. | How loudly the adapter speaks over the base model. Convention: set `α = r` or `α = 2r`. Larger α → stronger task signal, more risk of dominating. |
| `target_modules` | Which layers get an adapter at all. | For decoder-only models: `["q_proj","k_proj","v_proj","o_proj"]` at minimum. For T5: `["q","v"]` is the standard light-weight choice. |
| `lora_dropout` | Dropout inside the adapter path. | 0.05–0.1 helps with overfitting on small datasets. Drop to 0 if your dataset is large. |

### Rank → params table

For a hidden dimension d = 1024:

| Rank r | LoRA params/layer (2·d·r) | % of full W (d·d) | Effective scale (α=2r) |
|---|---|---|---|
| 4  | 8 192      | 0.78% | 2.0× |
| 8  | 16 384     | 1.56% | 2.0× |
| 16 | 32 768     | 3.13% | 2.0× |
| 32 | 65 536     | 6.25% | 2.0× |
| 64 | 131 072    | 12.5% | 2.0× |

Bump rank from 8 to 32 — trainable-parameter count grows ×4 per layer. The effective scale stays constant if you keep α/r fixed, but the *capacity* of the adapter grows linearly with r.

---

## 5. Causal vs Seq2Seq — Two Architectures, Two Formats

> *SmolLM2 is a decoder. Flan-T5 is an encoder-decoder. Same goal, very different plumbing — and very different data format.*

The assignment asks you to fine-tune *two* models with very different shapes. The model you choose dictates how you format every training example, which trainer you use, and which target modules carry LoRA.

```
CAUSAL LM — SmolLM2                      SEQ2SEQ — Flan-T5
─────────────────────                    ───────────────────
                                            ┌─────────┐    ┌─────────┐
   ┌──────────────────┐                     │ encoder │ →  │ decoder │
   │ DECODER (L→R)    │                     └─────────┘    └─────────┘
   └──────────────────┘
                                          INPUT:              TARGET:
   INPUT (one string):                    "What is X?         "X is like..."
                                          level: child"
   ### Question: What is X?
   ### Expertise level: child
   ### Answer: X is like ...
                                          loss only on
   loss on every token,                   target tokens
   predict next token
```

### Decoder-only: one string, next-token prediction

A causal LM like `SmolLM2-360M-Instruct` is a single transformer stack that reads left to right. Every token attends only to tokens before it. Training a causal LM means: give it a sequence, and ask it to predict each next token. The loss is cross-entropy over every position.

For instruction tuning, you concatenate the question, level, and answer into one string with clear delimiters. The model is then trained to produce the entire string — but in practice, the parts before the answer are essentially "free hints" that condition the generation.

→ Use the TRL library's `SFTTrainer`, which wraps everything as next-token prediction. Pass it a dataset with one `text` field per example.

### Encoder-decoder: input → target

Flan-T5 has two stacks. The *encoder* reads the full input (question + level) bidirectionally. The *decoder* generates the answer one token at a time, attending to the encoder's representation through cross-attention. The training loss applies only to the decoder's target tokens.

Crucially: the input and target are *separate columns*. You don't concatenate them. The trainer's data collator handles padding, label masking, and the decoder-input shift internally.

→ Use HuggingFace's `Seq2SeqTrainer` with `DataCollatorForSeq2Seq`. Dataset has separate `input_ids` and `labels`.

### The format check

| Aspect | SmolLM2 (causal) | Flan-T5 (seq2seq) |
|---|---|---|
| Dataset field(s) | One `text` column | `input` + `target` |
| Trainer | `SFTTrainer` (from trl) | `Seq2SeqTrainer` (transformers) |
| LoRA `task_type` | `TaskType.CAUSAL_LM` | `TaskType.SEQ_2_SEQ_LM` |
| Typical `target_modules` | `q_proj, k_proj, v_proj, o_proj` | `q, v` |
| Loss applied to | Every token in the string | Only the target tokens |

> ⚠️ **The single most common mistake:** feeding the seq2seq-style two-column dataset to `SFTTrainer`, or feeding the concatenated single-string dataset to `Seq2SeqTrainer`. The trainer will not always error — it may "train" and produce nonsense outputs. Always build two distinct dataset variants and use the right trainer for each.

---

## 6. Building the Dataset (Task 1.1)

> *580 filtered questions. Three expertise levels. One synthetic generation pass. One test set you review yourself.*

The pre-filtered dataset gives you 580 Q&A pairs. You need each question answered three times — once for each expertise level. The standard recipe is:

1. **Prompt a teacher model** (any capable LLM API, or a larger local model) with a meta-prompt: "Rewrite this answer for a child / student / expert." Request JSON output.
2. **Parse and validate** each response. Reject any that fail JSON parsing or that omit a level.
3. **Save as JSONL** with one example per line. JSONL is loadable, line-by-line, by HuggingFace `datasets`.
4. **Hand-pick 20 examples** for the test set. These come from the same Dolly source but are *excluded* from training. Manually review them.

### What "level" actually means

| Level | Voice cues | Allowed jargon | Typical length |
|---|---|---|---|
| **Child** | Analogies, everyday objects, simple verbs, no symbols | None — translate every term | 2–3 short sentences |
| **Student** | Definition + intuition + small example | Field-standard terms with brief gloss | 4–6 sentences, light structure |
| **Expert** | Precise definitions, technical vocabulary, no hand-holding | All of it | Compact and dense |

### Two dataset variants

Build the dataset once (the JSONL with question + level + answer per row), then derive the two trainer-specific formats:

```json
// Causal LM variant — flat text field
{
  "text": "### Question: Why is the sky blue?\n### Expertise level: child\n### Answer: The air bends blue light more than red light, so when you look up the sky looks blue."
}

// Seq2Seq variant — input/target columns
{
  "input":  "Question: Why is the sky blue? Expertise level: child",
  "target": "The air bends blue light more than red light, so when you look up the sky looks blue."
}
```

> 💡 **Sanity-check before you generate all 580:** generate 5 examples first. Verify: (1) JSON parses, (2) each question yields all three levels, (3) the saved JSONL loads back with `datasets.load_dataset("json", data_files=...)`, (4) the model tokenizes the text without truncating. Only then scale to the full set.

### How many examples do you really need?

For a behavior-shaping task like this, ~300–500 examples per level (1000–1500 total) is the practical floor where you start seeing reliable level adaptation. The full 580 × 3 = 1740 is more than enough. If you are rate-limited on the teacher model, start with a subset — say 150 × 3 = 450 — and check if it works before generating more.

---

## 7. The Training Recipe

> *Six objects, in this order. Get the order wrong and you will train nothing.*

### The canonical order of operations

1. **Tokenizer** — load, set `pad_token` if missing.
2. **BitsAndBytesConfig** — the 4-bit recipe.
3. **Base model** — load with `quantization_config=bnb_config`.
4. **prepare_model_for_kbit_training** — wraps the model to make quantized training stable.
5. **LoraConfig** — describe the adapter.
6. **Trainer** — pass it the model, dataset, and `peft_config`. Train.

### The key training arguments

| Argument | Sensible default | What it does |
|---|---|---|
| `num_train_epochs` | 1–3 | Passes through the full dataset. For ~1500 examples, 3 epochs is plenty. |
| `per_device_train_batch_size` | 4–8 | How many examples per GPU step. Limit by memory. |
| `gradient_accumulation_steps` | 2–4 | Effective batch = batch_size × accumulation. Used when memory caps batch_size low. |
| `learning_rate` | 2e-4 (causal), 1e-3 (T5) | LoRA tolerates higher LRs than full FT. T5 prefers larger LRs. |
| `warmup_ratio` | 0.03 | Ramp the LR up over the first 3% of training. Stops early instability. |
| `max_seq_length` | 512–1024 | Longer than your longest example; shorter than what fits in VRAM. |
| `logging_steps` | 10 | Print loss every 10 steps so you can watch it descend. |

### What "trainable parameters" should look like

After you call `model.print_trainable_parameters()`, you should see something like:

```
trainable params: 786,432 || all params: 362,003,584 || trainable%: 0.2173
```

If **trainable%** is < 0.01%, your LoRA didn't attach (check `target_modules` matches actual module names). If it is > 5%, something is mis-frozen.

> ⚠️ **Gradient flowing into nothing:** if your loss prints as `nan` from step 0, the LoRA adapter is sitting on a frozen tensor and the gradient has nowhere to go. The fix is almost always: don't apply `prepare_model_for_kbit_training` *after* calling `get_peft_model`. The PEFT step should come *after* kbit-prep.

---

## 8. Evaluation — Reading Two Generations Side by Side

> *Run each test question through both the base and the fine-tuned model. Look at them side by side. Score with intent, not vibes.*

The evaluation table demands four columns of judgment per example. Be deliberate about each.

| Column | What you're asking |
|---|---|
| **Level match** | Did the model actually use child-level / student-level / expert-level voice? A "child" answer with the word "Bayesian" fails this. |
| **Clarity / relevance** | Is the answer about the question, in plain readable English? Or did it wander? |
| **Faithfulness** | Does it stay anchored to the source answer, or invent claims? (Hallucination check.) |
| **Better after FT?** | Comparing base vs FT for this example: yes / no / tie. Be honest — fine-tuning sometimes regresses on individual examples. |

### Reproducible generation

Use the same generation arguments for both models. Otherwise you are comparing prompting styles, not models.

```python
gen_kwargs = dict(
    max_new_tokens=200,
    do_sample=False,        # greedy → reproducible
    num_beams=1,
    repetition_penalty=1.05,
)
```

For the base model, you build the same prompt format you trained on (or for T5, the same input column). The trick is: **the base model has never seen your delimiters before**, so its outputs may be off-format. That is part of the comparison — fine-tuning teaches the model to obey the format.

### What to look for in the discussion (Task 1.4)

The two architectures fail differently. Things to watch for:

- **SmolLM2** tends to keep generating until it hits `max_new_tokens`, sometimes "explaining the next question" you didn't ask. Stopping cleanly is a learned behavior.
- **Flan-T5-small** tends to be terse — sometimes too terse for "expert" answers. Its encoder-decoder structure gives crisper alignment with the question but smaller output vocabulary use.
- **Hallucination** shows up as confident factual errors. The base model invents, the fine-tuned one usually inherits from your training data — so the training data quality matters.

---

## 9. Common Failure Modes

A short field guide to the errors you will probably hit. Read once before training so you recognize them.

**1. Loss stays flat at the initial value.** The LoRA adapter is probably attached but not getting gradient. Causes: wrong order of `prepare_model_for_kbit_training` and `get_peft_model`; or `model.gradient_checkpointing_enable()` wasn't followed by `model.enable_input_require_grads()`; or `target_modules` doesn't match real module names.

**2. Loss is NaN.** Either your compute dtype is fp16 (use bf16 if your hardware supports it; T4 GPUs do), or your learning rate is far too high, or your tokenizer is producing degenerate sequences (all pad tokens).

**3. Tokenizer has no pad_token.** Most causal-LM tokenizers ship without a pad token. Set `tokenizer.pad_token = tokenizer.eos_token` before training. Also set `tokenizer.padding_side = "right"` for training, switch to `"left"` for inference.

**4. The fine-tuned model produces nothing different.** The adapter trained but doesn't get loaded. After `trainer.train()`, save with `model.save_pretrained(path)`. To use, load the base model again, then attach the saved adapter with `PeftModel.from_pretrained(base, path)`.

**5. Out of memory on a free GPU.** Reduce `per_device_train_batch_size`, then raise `gradient_accumulation_steps` to keep effective batch size constant. Lower `max_seq_length` to your real max length. Turn on `gradient_checkpointing=True`.

**6. Outputs in the wrong format after fine-tuning.** You trained too few epochs, or your delimiters were inconsistent between examples (sometimes `### Answer:`, sometimes `Answer:`). Be ruthlessly consistent.

---

## 10. Task Cards

Each task with concepts and questions to think about. No code — that part is yours.

### Task 1.1 — Prepare the level-adaptive dataset (15 pts)

**Concepts needed:**
- JSON / JSONL format and the HuggingFace `datasets.load_dataset("json", ...)` loader
- Calling an LLM API (or a local model) for synthetic data generation
- The difference between causal-LM string format and seq2seq input/target format (§ 5)

**Thinking:**
- What meta-prompt produces all three levels reliably in one call? (One JSON object with three fields is cleaner than three calls.)
- How will you guarantee the test set is *not* in the training set? Split *before* generating, or hold out specific question indices.
- What's your manual-review rubric for the 20 test examples?

### Task 1.2 — QLoRA fine-tune both models (10 pts)

**Concepts needed:**
- `BitsAndBytesConfig` with NF4 + double quantization (§ 3)
- `LoraConfig` with appropriate `task_type` and `target_modules` for each architecture (§ 4, § 5)
- The trainer match: `SFTTrainer` for causal, `Seq2SeqTrainer` for T5 (§ 5)
- `prepare_model_for_kbit_training` and its ordering relative to `get_peft_model` (§ 7)

**Thinking:**
- Choose r, alpha, and target_modules for each model. Why those choices?
- Sanity check: what should `print_trainable_parameters()` show? If trainable% is < 0.01% or > 5%, something is wrong.
- Save both adapters with clear names. You will reload them in Task 1.3.

### Task 1.3 — Evaluation table — before vs after (20 pts)

**Concepts needed:**
- Loading a base model + a PEFT adapter for inference (`PeftModel.from_pretrained`)
- Greedy generation with consistent `gen_kwargs` across runs (§ 8)
- The same prompt format used during training — for the base model too

**Thinking:**
- Generate four outputs per question: base-SmolLM2, FT-SmolLM2, base-T5, FT-T5. That's 80 total — automate the loop.
- What does "better after fine-tuning?" mean when both are bad? When both are good but stylistically different?
- Spot-check 2–3 examples manually before trusting the table.

### Task 1.4 — Model comparison and discussion (15 pts)

**Concepts needed:**
- Decoder-only vs encoder-decoder behavior differences (§ 5)
- How model size (360M vs ~77M for T5-small) affects expressiveness
- What "hallucination" looks like differently in each architecture

**Thinking:**
- Which architecture adapted faster — and is "faster" the same as "better"?
- Was one model better at one level (e.g., expert) and worse at another (child)?
- Tie observations to specific examples from your Task 1.3 table — cite quotes.

### Task 1.5 — Conceptual questions (4 questions, 15 pts)

**Concepts needed:**
- Q1 — What changes after FT: instruction-following and format compliance, not factual knowledge (§ 1, § 8)
- Q2 — Why QLoRA saves memory: 4-bit base storage + LoRA gradient on ~0.2% of params (§ 2, § 3)
- Q3 — Higher rank trade-off: capacity vs memory vs overfitting (§ 4)
- Q4 — LoRA vs prompting: where each wins (§ 1)

**Thinking:**
- For Q3: if your data is small (1500 examples), what r is appropriate? When would you go higher?
- For Q4: prompting wins when... fine-tuning wins when... cite the assignment's task as an example of which.

---

*Guide 03 · QLoRA Fine-Tuning · LLM Architectures Hometask 3 · Part 1*
