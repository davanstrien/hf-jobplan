# hf-jobplan

Experimental `hf` CLI extension. `hf jobplan <model>` prints a short prompt for running that model on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs): fine-tuning, embedding or bulk inference. It launches nothing.

```
hf extensions install davanstrien/hf-jobplan
hf jobplan answerdotai/ModernBERT-base
hf jobplan thinkingmachines/Inkling-2-Mini --task train-llm
hf jobplan --list
```

## How it works

One mapping, in [`src/hf_jobplan/prompts.py`](src/hf_jobplan/prompts.py):

1. The model's task tag on the Hub picks a default workload. An encoder (`fill-mask`) gets `train-encoder`, an embedding model gets `embed`, an LLM gets `generate`. `--task` picks another one.
2. The workload picks the prompt: one pointer for that workload, and the folder of ready [uv scripts](https://huggingface.co/uv-scripts) to read first.
3. The hints in the prompt are existing `hf` commands (`hf models card`, `hf jobs hardware`, `hf jobs uv run --help`, ...). The agent reads live data with the CLI it already has. The extension makes one Hub API call and keeps no copy of prices, flavors or model cards.

The pointer names the library to use:

| Workload | Pointer |
|---|---|
| `train-llm` | [TRL](https://huggingface.co/docs/trl/sft_trainer) `SFTTrainer` |
| `train-encoder` | transformers `Trainer` for a classifier, [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html) for an embedding model |
| `embed` | Sentence Transformers, or Text Embeddings Inference |
| `generate`, `ocr` | the model's server (vLLM or SGLang) inside the Job, with concurrent requests |
| `classify`, `transcribe` | the library and loading code from the model card |

## Example

`hf jobplan answerdotai/ModernBERT-base` prints:

````
# answerdotai/ModernBERT-base on Hugging Face Jobs: fine-tune an encoder or an embedding model

Kind of model: task tag `fill-mask`, library `transformers`. Workload: `train-encoder` (the default for this task tag).
Other workloads: `hf jobplan answerdotai/ModernBERT-base --task <generate, ocr, embed, classify, transcribe, train-llm>`. The whole mapping: `hf jobplan --list`.

## Find out first (each line is an existing command)
    hf models card answerdotai/ModernBERT-base --text
        how the authors run it: library or server, container image, flags
    hf models ls answerdotai/ModernBERT-base
        weight files and sizes: the GPU memory you need
    hf endpoints catalog ls --search ModernBERT-base
        servers tested for this model, if it is listed
    hf datasets ls uv-scripts/classification
        ready uv scripts for this workload: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image

## For this workload
Use the transformers `Trainer` for a classifier, or Sentence Transformers for an embedding model (https://sbert.net/docs/sentence_transformer/training_overview.html). The Job disk is deleted at the end: save the model to a mounted bucket or a private repo.

## Then
1. Write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first (a few steps on a few rows). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write to the place the user named: mount a bucket with `-v` (see the help), or see `hf buckets --help`.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, the evaluation result, and what you did not check.
````

## The mapping

`hf jobplan --list` prints:

```
Hub task tag -> default workload

  feature-extraction             embed
  sentence-similarity            embed
  text-ranking                   embed
  fill-mask                      train-encoder
  text-classification            classify
  token-classification           classify
  zero-shot-classification       classify
  image-classification           classify
  object-detection               classify
  text-generation                generate
  image-text-to-text             generate
  any-to-any                     generate
  automatic-speech-recognition   transcribe
  (a name or tag with `ocr`)     ocr
  (any other tag)                generate

Workloads

  generate       bulk generation, classification or extraction with an LLM or VLM; scripts: uv-scripts/vllm
  ocr            pages to text with an OCR model; scripts: uv-scripts/ocr
  embed          embed a text column; scripts: uv-scripts/embeddings
  classify       batch inference with a classifier; scripts: uv-scripts/classification
  transcribe     transcribe audio files; scripts: uv-scripts/transcription
  train-llm      fine-tune an LLM; scripts: uv-scripts/training
  train-encoder  fine-tune an encoder or an embedding model; scripts: uv-scripts/classification
```

## Why

An agent that reads the right script or model card writes correct Job commands. The hard part is finding the right path for a model: which library or server, which image, which existing script. A prompt on a model page can then be one line:

> Run `hf jobplan <model>` and follow what it prints.

## Status

Prototype. The output format and the workload names will change. `hf-jobplan` in the repo root is the first single-file prototype, kept while tests that call it are running.

## Develop

```
uv run pytest
uv run ruff check .
```
