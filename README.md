# hf-jobplan

Experimental `hf` CLI extension. `hf jobplan <model>` prints what an agent (or a person) needs to run a model on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs). It launches nothing.

```
hf jobplan lightonai/LightOnOCR-3-0.8B
hf jobplan answerdotai/ModernBERT-base --task train-encoder
```

## What it prints

- **Facts** from the Hub API: task tag, library, weight size, whether the repo runs custom code.
- **How the card says to run it**: the install, serve and docker code blocks of the model card.
- **Tested serving engines** for the model in the Inference Endpoints catalogue.
- **uv scripts that name the model** in <https://huggingface.co/uv-scripts>.
- **Flavors that fit the weights**, cheapest first, from `hf jobs hardware`.
- **One pointer for the task** (`generate`, `ocr`, `embed`, `train-llm`, `train-encoder`). The default task comes from the model's task tag.
- **A procedure**: one uv script, a few rows first, pick a flavor and say why, full run, report.

## Why

An agent that reads the right script or model card writes correct Job commands. The hard part is finding the right path for a model: which server, which image, which existing script. This command does that search in one deterministic step, so the prompt on a model page can stay short:

> Run `hf jobplan <model>` and follow what it prints.

## Status

Prototype. Output format and task names will change. The repo keeps the first single-file prototype (`hf-jobplan`, a uv script) next to the package while tests that call it are running.

## Develop

```
uv run pytest
uv run ruff check .
```
