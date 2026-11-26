# hf-jobplan

Experimental `hf` CLI extension. `hf jobplan <model>` prints a short prompt for running or fine-tuning that model on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs). It launches nothing.

```
hf extensions install davanstrien/hf-jobplan
hf jobplan answerdotai/ModernBERT-base            # an encoder: the default verb is train
hf jobplan google/embeddinggemma-2                # an embedding model: the default verb is infer
hf jobplan thinkingmachines/Inkling-2-Mini --task train
hf jobplan thinkingmachines/Inkling-2-Mini --short   # the one-sentence prompt
hf jobplan --list
```

## The short prompt

`--short` prints one sentence, 44 to 56 words, to paste into an agent or to show on a model page. You fill in the dataset and the bucket. It keeps only the clauses that changed what agents did in tests: the pointer for the kind of model, the bucket mount, a small test first, and the report.

`hf jobplan thinkingmachines/Inkling-2-Mini --short`:

> Run thinkingmachines/Inkling-2-Mini over <dataset> on Hugging Face Jobs with the `hf` CLI: serve it with vLLM or SGLang inside the Job and send concurrent requests, write to the bucket mounted with `-v hf://buckets/<bucket>:/out` (a token only for gated or private repos), test a few rows first, and report the Job ids and cost per 1,000 items.

`hf jobplan answerdotai/ModernBERT-base --short`:

> Fine-tune answerdotai/ModernBERT-base on <dataset> on Hugging Face Jobs with the `hf` CLI: use the transformers Trainer, write to the bucket mounted with `-v hf://buckets/<bucket>:/out` (a token only for gated or private repos), test a few rows first, and report the Job ids, cost and eval result.

For a gated model the token clause changes to "this model is gated: pass `--secrets HF_TOKEN`".

## How it works

One mapping, in [`src/hf_jobplan/prompts.py`](src/hf_jobplan/prompts.py):

1. The model's task tag on the Hub gives the kind of model (`llm`, `embedding`, `encoder`, `classifier`, `ocr`, `speech`, `diffusion`). A tag that is not in the mapping gives the kind `other`.
2. There are two verbs, `infer` and `train`. An encoder defaults to `train`. Every other kind defaults to `infer`. `--task` picks the other one.
3. The verb and the kind pick one pointer: the library or server to use. It is the first thing in the prompt.
4. The rest of the prompt is the same for every model. Its hints are existing `hf` commands (`hf models card`, `hf jobs hardware`, `hf jobs uv run --help`, ...), so the agent reads live data with the CLI it already has. The extension makes one Hub API call and keeps no copy of prices, flavors or model cards.
5. Outputs go to a bucket mounted into the Job (`-v hf://buckets/<namespace>/<bucket>:/out`), so the Job needs no token to save its results.

Every model gets a prompt. A kind without its own pointer uses the `other` pointer for the verb ("use the library and the loading code from the model card"). If the Hub lookup fails, the command prints the same prompt without the facts line.

| Verb | Kind | Pointer |
|---|---|---|
| `train` | `llm` | [TRL](https://github.com/huggingface/trl): its ready [`sft.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) script runs on Jobs as it is ([guide](https://huggingface.co/docs/trl/jobs_training)) |
| `train` | `encoder`, `classifier` | [transformers](https://github.com/huggingface/transformers) [`Trainer`](https://huggingface.co/docs/transformers/trainer) |
| `train` | `embedding` | [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html), with an evaluator before and after |
| `train` | `diffusion` | the [diffusers training scripts](https://huggingface.co/docs/diffusers/training/overview) |
| `infer` | `llm`, `ocr` | the model's server (vLLM or SGLang) inside the Job, with concurrent requests; [`saturate`](https://github.com/davanstrien/saturate) as the client |
| `infer` | `embedding` | Sentence Transformers, or Text Embeddings Inference |
| `infer` | `classifier` | transformers `pipeline` |
| `infer` | `diffusion` | the diffusers pipeline from the model card |
| either | `other` | what the model card documents |

## Example: an encoder

`hf jobplan answerdotai/ModernBERT-base` prints:

````
# Fine-tune answerdotai/ModernBERT-base on Hugging Face Jobs

Everything below uses the `hf` CLI, which printed this. `hf --help` lists its commands, and every command takes `--help`.

Task tag `fill-mask`, library `transformers` -> kind `encoder`. Verb: `train` (the default for this kind).
Docs as markdown: Jobs https://huggingface.co/docs/hub/jobs.md; `transformers` index https://huggingface.co/docs/transformers/llms.txt. Any Hugging Face docs page is markdown if you add `.md` to its URL.
The other verb: `hf jobplan answerdotai/ModernBERT-base --task infer`. The whole mapping: `hf jobplan --list`.

## For this kind of model
Use the transformers `Trainer` (https://huggingface.co/docs/transformers/trainer.md, source https://github.com/huggingface/transformers) for a classifier, or Sentence Transformers (https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/sentence_transformer/training_overview.md) for an embedding model. transformers has no training script to run by URL: write a uv script, or read `train-classifier.py` in the scripts folder above. The Job disk is deleted at the end: write checkpoints and the final model to the mounted bucket (step 4).

## Find out first (each line is an existing command)
    hf models card answerdotai/ModernBERT-base --text
        how the authors run it: library or server, container image, flags
    hf models info answerdotai/ModernBERT-base
        task, library, gated or not, parameter count and dtype (the GPU memory you need), files
    hf endpoints catalog ls --search ModernBERT-base
        servers tested for this model, if it is listed
    hf datasets ls uv-scripts/classification
        ready uv scripts: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image

## Then
1. Use a ready script if one fits. If not, write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first (a few steps on a few rows). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write outputs and checkpoints to a bucket mounted into the Job: `-v hf://buckets/<namespace>/<bucket>:/out`, then write under `/out`. A bucket mount is read and write, so the Job needs no token to save its results. Create the bucket first with `hf buckets create <namespace>/<bucket>` if it does not exist. A dataset mounts the same way, read-only: `-v hf://datasets/<org>/<dataset>:/data`. Pass `--secrets HF_TOKEN` only when the Job must read a gated or private model or dataset, or when the user asked for a model or dataset repo as the result.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, the evaluation result, and what you did not check.
````

## Example: the fallback

`hf jobplan autogluon/mitra-classifier` has a task tag that is not in the mapping. The top of the output and the pointer:

````
# Run autogluon/mitra-classifier on Hugging Face Jobs: batch inference

Everything below uses the `hf` CLI, which printed this. `hf --help` lists its commands, and every command takes `--help`.

Task tag `tabular-classification`, library `None` -> kind `other`. Verb: `infer` (the default for this kind).
...
## For this kind of model
Use the library and the loading code from the model card. If the card has none, the library name above and the files in the repo show how to load the model.
...
````

## The mapping

`hf jobplan --list` prints:

```
Hub task tag -> kind of model

  text-generation                llm
  image-text-to-text             llm
  any-to-any                     llm
  feature-extraction             embedding
  sentence-similarity            embedding
  text-ranking                   embedding
  fill-mask                      encoder
  text-classification            classifier
  token-classification           classifier
  zero-shot-classification       classifier
  image-classification           classifier
  object-detection               classifier
  automatic-speech-recognition   speech
  text-to-image                  diffusion
  image-to-image                 diffusion
  text-to-video                  diffusion
  image-to-video                 diffusion
  (an OCR tag or name, see below) ocr
  (any other tag, or no tag)     other

OCR tags: document-ai, document-parse, document-parsing, document-understanding, handwriting-recognition, ocr, pdf2markdown; or `ocr` / `docling` in the model id or a tag.

Docs index per library: transformers, trl, peft, diffusers, timm, sentence-transformers

Default verb: train for encoder; infer for every other kind.

(verb, kind) -> pointer

  infer  llm         Start the model's server (vLLM or SGLang, whichever the model card or the catalogue names) inside the Job and send many requests at once to localhost.
  infer  ocr         Follow the model card.
  infer  embedding   Use Sentence Transformers with a large batch size and the prompt names from the model card, or Text Embeddings Inference if the catalogue lists the `tei` engine.
  infer  classifier  Use the transformers `pipeline` or the loading code from the model card, with batches on one small GPU.
  infer  speech      Use the library from the model card.
  infer  diffusion   Use the diffusers pipeline from the model card (https://huggingface.co/docs/diffusers/index.md).
  infer  other       Use the library and the loading code from the model card.
  train  llm         Use TRL (https://github.com/huggingface/trl).
  train  ocr         Use TRL (https://github.com/huggingface/trl).
  train  embedding   Use Sentence Transformers (https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/sentence_transformer/training_overview.md) and run an evaluator before and after.
  train  encoder     Use the transformers `Trainer` (https://huggingface.co/docs/transformers/trainer.md, source https://github.com/huggingface/transformers) for a classifier, or Sentence Transformers (https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/sentence_transformer/training_overview.md) for an embedding model.
  train  classifier  Use the transformers `Trainer` (https://huggingface.co/docs/transformers/trainer.md, source https://github.com/huggingface/transformers).
  train  diffusion   Use the diffusers training scripts (overview: https://huggingface.co/docs/diffusers/training/overview.md, LoRA: https://huggingface.co/docs/diffusers/training/lora.md, scripts: https://github.com/huggingface/diffusers/tree/main/examples).
  train  other       Use the training method that the model card or its library documents.
```

## Why

An agent that reads the right script or model card writes correct Job commands. The hard part is finding the right path for a model: which library or server, which image, which existing script. A prompt on a model page can then be one line:

> Run `hf jobplan <model>` and follow what it prints.

## Status

Prototype. The output format, the names and the wording will change.

What has been tried so far (2026-11-26). The agents were clean headless sessions that got only the prompt. One trial per cell.

**The long output (`hf jobplan <model>`), plan-only.** The agent writes the script and the exact commands and launches nothing. 7 model and workload cases, with Claude Haiku, Sonnet and Opus.

- All 42 agents installed the extension and ran it.
- After the bucket-mount step was added, 0 of 21 plans passed a token into the Job. With the text before it, 12 of 21 did.
- With the `saturate` clause, Sonnet used it in 3 of 3 bulk-inference plans and Opus in 2 of 3. Haiku used it in 0 of 3.
- Two open models in another agent harness (GLM-5.3 and DeepSeek-V4.1-Flash in pi) also installed it in 8 of 8 runs, and 7 of the 8 wrote a complete plan.

**A short prompt with the same three ideas, plan-only.** The wording was close to `--short`, with the token rule as a full sentence. Same 7 cases and 3 agents, 21 runs, compared with a bare request ("do this task with this model on Jobs").

- A server for bulk LLM and OCR inference: 6 of 6 plans, against 0 of 6 for the bare request (those used vLLM in-process).
- A token passed into the Job: 1 of 21 plans, against 17 of 21 for the bare request.
- The expected training library: 9 of 9 plans, the same as the bare request.

**Real runs on Jobs.** Six agents (Sonnet and Opus) classified 5,000 texts with `thinkingmachines/Inkling-2-Mini`, from a longer prompt with the same pointer and bucket lines, not from this extension.

- All six finished. The four that served the model inside the Job passed no token and wrote their results through the bucket mount.
- Cost was $0.07 to $0.08 per 1,000 texts with a server inside the Job, and $0.11 to $0.13 with the model on an Inference Endpoint.
- `saturate` against a hand-written client, same endpoint and hardware: 22.5 against 17.8 texts per second, one run each.
- No agent found the request form that stops the model from reasoning before it answers. That made every run several times slower than a tuned run of the same model.

Limits: one trial per cell; a plan-only run does not show that a plan works; the exact `--short` text and the current section order of the long output have no finished test yet; the real runs cover one model and one task.

## Develop

```
uv run pytest
uv run ruff check .
```
