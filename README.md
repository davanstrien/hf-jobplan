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
| `train` | `llm` | [TRL](https://huggingface.co/docs/trl/sft_trainer) `SFTTrainer` |
| `train` | `encoder`, `classifier` | [transformers](https://huggingface.co/docs/transformers/training) `Trainer` |
| `train` | `embedding` | [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html), with an evaluator before and after |
| `infer` | `llm`, `ocr` | the model's server (vLLM or SGLang) inside the Job, with concurrent requests |
| `infer` | `embedding` | Sentence Transformers, or Text Embeddings Inference |
| `infer` | `classifier` | transformers `pipeline` |
| either | `other` | what the model card documents |

## Example: an encoder

`hf jobplan answerdotai/ModernBERT-base` prints:

````
# Fine-tune answerdotai/ModernBERT-base on Hugging Face Jobs

Task tag `fill-mask`, library `transformers` -> kind `encoder`. Verb: `train` (the default for this kind).
The other verb: `hf jobplan answerdotai/ModernBERT-base --task infer`. The whole mapping: `hf jobplan --list`.

## Find out first (each line is an existing command)
    hf models card answerdotai/ModernBERT-base --text
        how the authors run it: library or server, container image, flags
    hf models ls answerdotai/ModernBERT-base
        weight files and sizes: the GPU memory you need
    hf endpoints catalog ls --search ModernBERT-base
        servers tested for this model, if it is listed
    hf datasets ls uv-scripts/classification
        ready uv scripts: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image

## For this kind of model
Use the transformers `Trainer` (https://huggingface.co/docs/transformers/training) for a classifier, or Sentence Transformers (https://sbert.net/docs/sentence_transformer/training_overview.html) for an embedding model. The Job disk is deleted at the end: save checkpoints and the final model to a mounted bucket or a private repo.

## Then
1. Write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first (a few steps on a few rows). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write to the place the user named: mount a bucket with `-v` (see the help), or see `hf buckets --help`.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, the evaluation result, and what you did not check.
````

## Example: the fallback

`hf jobplan black-forest-labs/FLUX.1-dev` has a task tag that is not in the mapping. The top of the output and the pointer:

````
# Run black-forest-labs/FLUX.1-dev on Hugging Face Jobs: batch inference

Task tag `text-to-image`, library `diffusers` -> kind `other`. Verb: `infer` (the default for this kind).
The other verb: `hf jobplan black-forest-labs/FLUX.1-dev --task train`. The whole mapping: `hf jobplan --list`.
...
Use the library and the loading code from the model card. If the card has none, the library name above and the files in the repo show how to load the model.
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
  (a name or tag with `ocr`)     ocr
  (any other tag, or no tag)     other

Default verb: train for encoder; infer for every other kind.

(verb, kind) -> pointer

  infer  llm         Start the model's server (vLLM or SGLang, whichever the model card or the catalogue names) inside the Job and send many requests at once to localhost.
  infer  ocr         Start the model's server (vLLM or SGLang, whichever the model card names) inside the Job and send one request per page to localhost, many at once.
  infer  embedding   Use Sentence Transformers with a large batch size and the prompt names from the model card, or Text Embeddings Inference if the catalogue lists the `tei` engine.
  infer  classifier  Use the transformers `pipeline` or the loading code from the model card, with batches on one small GPU.
  infer  speech      Use the library from the model card.
  infer  other       Use the library and the loading code from the model card.
  train  llm         Use TRL `SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer).
  train  ocr         Use TRL `SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer) or the fine-tuning code the model card links.
  train  embedding   Use Sentence Transformers (https://sbert.net/docs/sentence_transformer/training_overview.html) and run an evaluator before and after.
  train  encoder     Use the transformers `Trainer` (https://huggingface.co/docs/transformers/training) for a classifier, or Sentence Transformers (https://sbert.net/docs/sentence_transformer/training_overview.html) for an embedding model.
  train  classifier  Use the transformers `Trainer` (https://huggingface.co/docs/transformers/training).
  train  other       Use the training method that the model card or its library documents.
```

## Why

An agent that reads the right script or model card writes correct Job commands. The hard part is finding the right path for a model: which library or server, which image, which existing script. A prompt on a model page can then be one line:

> Run `hf jobplan <model>` and follow what it prints.

## Status

Prototype. The output format and the names will change. `hf-jobplan` in the repo root is the first single-file prototype, kept while tests that call it are running.

## Develop

```
uv run pytest
uv run ruff check .
```
