# hf-jobplan

Experimental `hf` CLI extension. `hf jobplan <model>` prints a short prompt for running or fine-tuning that model on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs). It launches nothing.

```
hf extensions install davanstrien/hf-jobplan
hf jobplan answerdotai/ModernBERT-base            # an encoder: the default verb is train
hf jobplan google/embeddinggemma-2                # an embedding model: the default verb is infer
hf jobplan thinkingmachines/Inkling-2-Mini --task train
hf jobplan --list
```

## How it works

One mapping, in [`src/hf_jobplan/prompts.py`](src/hf_jobplan/prompts.py):

1. The model's task tag on the Hub gives the kind of model (`llm`, `embedding`, `encoder`, `classifier`, `ocr`, `speech`). A tag that is not in the mapping gives the kind `other`.
2. There are two verbs, `infer` and `train`. An encoder defaults to `train`. Every other kind defaults to `infer`. `--task` picks the other one.
3. The verb and the kind pick one pointer: the library or server to use.
4. The rest of the prompt is the same for every model. Its hints are existing `hf` commands (`hf models card`, `hf jobs hardware`, `hf jobs uv run --help`, ...), so the agent reads live data with the CLI it already has. The extension makes one Hub API call and keeps no copy of prices, flavors or model cards.

Every model gets a prompt. A kind without its own pointer uses the `other` pointer for the verb ("use the library and the loading code from the model card"). If the Hub lookup fails, the command prints the same prompt without the facts line.

| Verb | Kind | Pointer |
|---|---|---|
| `train` | `llm` | [TRL](https://github.com/huggingface/trl): its ready [`sft.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) script runs on Jobs as it is ([guide](https://huggingface.co/docs/trl/jobs_training)) |
| `train` | `encoder`, `classifier` | [transformers](https://github.com/huggingface/transformers) [`Trainer`](https://huggingface.co/docs/transformers/trainer) |
| `train` | `embedding` | [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html), with an evaluator before and after |
| `infer` | `llm`, `ocr` | the model's server (vLLM or SGLang) inside the Job, with concurrent requests |
| `infer` | `embedding` | Sentence Transformers, or Text Embeddings Inference |
| `infer` | `classifier` | transformers `pipeline` |
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
4. Run the full input. Write outputs and checkpoints to a bucket mounted into the Job: `-v hf://buckets/<namespace>/<bucket>:/out`, then write under `/out`. A bucket mount is read and write, so the Job needs no token to save its results. Create the bucket first with `hf buckets create <namespace>/<bucket>` if it does not exist. A dataset mounts the same way, read-only: `-v hf://datasets/<org>/<dataset>:/data`. Pass `--secrets HF_TOKEN` only when the Job must read a private or gated repo, or when the user asked for a model or dataset repo as the result.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, the evaluation result, and what you did not check.
````

## Example: the fallback

`hf jobplan autogluon/mitra-classifier` has a task tag that is not in the mapping. The top of the output and the pointer:

````
# Run autogluon/mitra-classifier on Hugging Face Jobs: batch inference

Everything below uses the `hf` CLI, which printed this. `hf --help` lists its commands, and every command takes `--help`.

...
Use the library and the loading code from the model card. If the card has none, the library name above and the files in the repo show how to load the model.

## Find out first (each line is an existing command)
    hf models card autogluon/mitra-classifier --text
        how the authors run it: library or server, container image, flags
    hf models info autogluon/mitra-classifier
        task, library, gated or not, parameter count and dtype (the GPU memory you need), files
    hf endpoints catalog ls --search mitra-classifier
        servers tested for this model, if it is listed
    hf datasets ls --author uv-scripts
        ready uv scripts: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image
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

Prototype. The output format and the names will change.

## Develop

```
uv run pytest
uv run ruff check .
```
