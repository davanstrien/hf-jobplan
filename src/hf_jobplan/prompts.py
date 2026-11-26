"""The mapping: Hub task tag -> kind of model; (verb, kind) -> prompt.

Everything the command prints comes from this file. Two verbs: `infer` and `train`. A model of
an unknown kind gets the `other` entries, so every model gets a prompt. Hints name existing
`hf` commands, so the agent reads live data with the CLI it already has.
"""

# Hub task tag (pipeline_tag) -> kind of model. A tag that is not here is kind `other`.
TASK_TAG_TO_KIND = {
    "text-generation": "llm",
    "image-text-to-text": "llm",
    "any-to-any": "llm",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "text-ranking": "embedding",
    "fill-mask": "encoder",
    "text-classification": "classifier",
    "token-classification": "classifier",
    "zero-shot-classification": "classifier",
    "image-classification": "classifier",
    "object-detection": "classifier",
    "automatic-speech-recognition": "speech",
    "text-to-image": "diffusion",
    "image-to-image": "diffusion",
    "text-to-video": "diffusion",
    "image-to-video": "diffusion",
}

# OCR and document-parsing models share task tags with general VLMs, so the task tag cannot find them.
# A model is kind `ocr` when one of its tags is in OCR_TAGS, or when its id or a tag contains one of OCR_WORDS.
OCR_TAGS = {"ocr", "document-parse", "document-parsing", "document-understanding", "pdf2markdown", "document-ai", "handwriting-recognition"}
OCR_WORDS = ("ocr", "docling")

# Docs an agent can read as text. Hugging Face docs serve each page as markdown (add `.md` to the URL)
# and an index of all pages at `llms.txt`. Keyed by the `library_name` of the model.
LIBRARY_DOCS = {
    "transformers": "https://huggingface.co/docs/transformers/llms.txt",
    "trl": "https://huggingface.co/docs/trl/llms.txt",
    "peft": "https://huggingface.co/docs/peft/llms.txt",
    "diffusers": "https://huggingface.co/docs/diffusers/llms.txt",
    "timm": "https://huggingface.co/docs/timm/llms.txt",
    "sentence-transformers": "https://github.com/huggingface/sentence-transformers/tree/main/docs",
}
JOBS_DOCS = "https://huggingface.co/docs/hub/jobs.md"

# Kinds that people mostly fine-tune. Every other kind defaults to `infer`.
TRAIN_BY_DEFAULT = {"encoder"}

SAVE = "The Job disk is deleted at the end: write checkpoints and the final model to the mounted bucket (step 4)."
SATURATE = "`saturate` (https://github.com/davanstrien/saturate) is a client for this: dataset in, concurrent requests to the server, resumable parquet out."
TRL = ("Use TRL (https://github.com/huggingface/trl). It has examples for some models: look in https://github.com/huggingface/trl/tree/main/examples "
       "for a folder that names this model or its family first. It also ships a general training script that runs on Jobs as it is: "
       "`hf jobs uv run --detach --timeout <limit> --flavor <flavor> -- https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py "
       "--model_name_or_path {repo} --dataset_name <dataset> --output_dir <dir>` (arguments: read the script; guide: https://huggingface.co/docs/trl/jobs_training.md; LoRA with PEFT: https://huggingface.co/docs/peft/quicktour.md). "
       "Write your own script with `SFTTrainer` only if the data needs code that the script does not have.")
TRAINER = "the transformers `Trainer` (https://huggingface.co/docs/transformers/trainer.md, source https://github.com/huggingface/transformers)"
SERVER = "Start the model's server (vLLM or SGLang, whichever the model card or the catalogue names) inside the Job and send many requests at once to localhost. A loop over `model.generate` is several times slower."

# (verb, kind) -> (what the run is, the one pointer that most changes cost or correctness, folder of ready uv scripts).
# (verb, "other") is the fallback for every kind without its own entry.
POINTERS = {
    ("infer", "llm"): ("bulk generation, classification or extraction", f"{SERVER} {SATURATE}", "uv-scripts/vllm"),
    ("infer", "ocr"): ("pages to text", f"Follow the model card. If it gives a vLLM or SGLang command, start that server inside the Job and send one request per page to localhost, many at once: a loop over pages is several times slower. {SATURATE} If the model has its own library (PaddleOCR, ONNX, ...), use that library with batches.", "uv-scripts/ocr"),
    ("infer", "embedding"): ("embed a dataset", "Use Sentence Transformers with a large batch size and the prompt names from the model card, or Text Embeddings Inference if the catalogue lists the `tei` engine.", "uv-scripts/embeddings"),
    ("infer", "classifier"): ("batch inference", "Use the transformers `pipeline` or the loading code from the model card, with batches on one small GPU.", "uv-scripts/classification"),
    ("infer", "speech"): ("transcribe audio", "Use the library from the model card. Sort the files by length before you batch them.", "uv-scripts/transcription"),
    ("infer", "diffusion"): ("batch generation", "Use the diffusers pipeline from the model card (https://huggingface.co/docs/diffusers/index.md). Load the pipeline once, send prompts in batches, and write each output file as it is made.", None),
    ("infer", "other"): ("batch inference", "Use the library and the loading code from the model card. If the card has none, the library name above and the files in the repo show how to load the model.", None),
    ("train", "llm"): ("fine-tune", f"{TRL} {SAVE}", "uv-scripts/training"),
    ("train", "ocr"): ("fine-tune", f"{TRL} For a vision model, check first that the script takes image data; the model card may link its own fine-tuning code. {SAVE}", "uv-scripts/transformers-training"),
    ("train", "embedding"): ("fine-tune", f"Use Sentence Transformers (https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/sentence_transformer/training_overview.md) and run an evaluator before and after. {SAVE}", "uv-scripts/embeddings"),
    ("train", "encoder"): ("fine-tune", f"Use {TRAINER} for a classifier, or Sentence Transformers (https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/sentence_transformer/training_overview.md) for an embedding model. transformers has no training script to run by URL: write a uv script, or read `train-classifier.py` in the scripts folder above. {SAVE}", "uv-scripts/classification"),
    ("train", "classifier"): ("fine-tune", f"Use {TRAINER}. transformers has no training script to run by URL: write a uv script, or read `train-classifier.py` in the scripts folder above. {SAVE}", "uv-scripts/classification"),
    ("train", "diffusion"): ("fine-tune", f"Use the diffusers training scripts (overview: https://huggingface.co/docs/diffusers/training/overview.md, LoRA: https://huggingface.co/docs/diffusers/training/lora.md, scripts: https://github.com/huggingface/diffusers/tree/main/examples). {SAVE}", None),
    ("train", "other"): ("fine-tune", f"Use the training method that the model card or its library documents. For most transformers models that is {TRAINER}. {SAVE}", None),
}

PROMPT = """# {verb_title} {repo} on Hugging Face Jobs{title}

Everything below uses the `hf` CLI, which printed this. `hf --help` lists its commands, and every command takes `--help`.

{facts}
{docs}
The other verb: `hf jobplan {repo} --task {other_verb}`. The whole mapping: `hf jobplan --list`.
{custom_code}
## For this kind of model
{pointer}

## Find out first (each line is an existing command)
    hf models card {repo} --text
        how the authors run it: library or server, container image, flags
    hf models info {repo}
        task, library, gated or not, parameter count and dtype (the GPU memory you need), files
    hf endpoints catalog ls --search {name}
        servers tested for this model, if it is listed
    {scripts}
        ready uv scripts: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image

## Then
1. Use a ready script if one fits. If not, write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first ({test}). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write outputs and checkpoints to a bucket mounted into the Job: `-v hf://buckets/<namespace>/<bucket>:/out`, then write under `/out`. A bucket mount is read and write, so the Job needs no token to save its results. Create the bucket first with `hf buckets create <namespace>/<bucket>` if it does not exist. A dataset mounts the same way, read-only: `-v hf://datasets/<org>/<dataset>:/data`. Pass `--secrets HF_TOKEN` only when the Job must read a gated or private model or dataset, or when the user asked for a model or dataset repo as the result.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, {report}, and what you did not check.
"""

CUSTOM_CODE = "\nThis repo runs its own code (`custom_code`). Do not pass your normal token or a secret to a Job that loads it.\n"


def kind_of(tag: str | None, repo: str, tags: list[str]) -> str:
    """OCR models by tag or name, the rest by task tag, `other` when the tag is unknown or missing."""
    lowered = [t.lower() for t in tags]
    if OCR_TAGS & set(lowered) or any(word in text for word in OCR_WORDS for text in [repo.lower(), *lowered]):
        return "ocr"
    return TASK_TAG_TO_KIND.get(tag or "", "other")


def render(repo: str, tag: str | None, library: str | None, tags: list[str], verb: str | None, found: bool = True, gated: bool = False) -> str:
    """The prompt for one model. `found=False` means the Hub lookup failed: print the fallback."""
    kind = kind_of(tag, repo, tags)
    chosen = verb or ("train" if kind in TRAIN_BY_DEFAULT else "infer")
    title, pointer, scripts = POINTERS.get((chosen, kind), POINTERS[(chosen, "other")])
    if not found:
        facts = "The Hub lookup for this model failed (private, gated, or a wrong id), so the kind of model is unknown. The steps below still apply."
    else:
        why = "" if verb else " (the default for this kind)"
        facts = f"Task tag `{tag}`, library `{library}` -> kind `{kind}`. Verb: `{chosen}`{why}."
    # A GGUF file or an adapter cannot be fine-tuned directly. The Hub puts the base model in a tag.
    derived = next((t.split(":", 2)[2] for t in tags if t.startswith(("base_model:quantized:", "base_model:adapter:"))), None)
    if chosen == "train" and derived:
        facts += f"\nThis repo is a quantized file or an adapter of `{derived}`. Fine-tune the base model: `hf jobplan {derived} --task train`."
    if gated:
        facts += (f"\nThis model is gated. Accept its terms at https://huggingface.co/{repo} first. The Job needs a token to download it: pass `--secrets HF_TOKEN`"
                  + (", and because the repo also runs its own code, use a fine-grained token that can only read this repo." if "custom_code" in tags else "."))
    library_docs = LIBRARY_DOCS.get((library or "").lower())
    docs = f"Docs as markdown: Jobs {JOBS_DOCS}" + (f"; `{library}` index {library_docs}" if library_docs else "") + ". Any Hugging Face docs page is markdown if you add `.md` to its URL."
    return PROMPT.format(
        docs=docs,
        repo=repo, name=repo.split("/")[-1], verb_title="Fine-tune" if chosen == "train" else "Run", title="" if chosen == "train" else f": {title}", facts=facts,
        other_verb="infer" if chosen == "train" else "train", custom_code=CUSTOM_CODE if "custom_code" in tags else "",
        scripts=f"hf datasets ls {scripts}" if scripts else "hf datasets ls --author uv-scripts", pointer=pointer.replace("{repo}", repo),
        test="a few steps on a few rows" if chosen == "train" else "a few rows",
        report="the evaluation result" if chosen == "train" else "cost per 1,000 items",
    )


# The short prompt: one sentence, with only the clauses that changed what agents did in tests.
# (verb, kind) -> the first clause. (verb, "other") is the fallback.
SHORT_POINTERS = {
    ("infer", "llm"): "serve it with vLLM or SGLang inside the Job and send concurrent requests",
    ("infer", "ocr"): "serve it with vLLM or SGLang inside the Job and send concurrent requests",
    ("infer", "embedding"): "use Sentence Transformers",
    ("infer", "classifier"): "use the transformers pipeline",
    ("infer", "other"): "use the library from the model card",
    ("train", "llm"): "use TRL",
    ("train", "ocr"): "use TRL",
    ("train", "embedding"): "use Sentence Transformers",
    ("train", "encoder"): "use the transformers Trainer",
    ("train", "classifier"): "use the transformers Trainer",
    ("train", "diffusion"): "use the diffusers training scripts",
    ("train", "other"): "use the training method from the model card",
}
SHORT = ("{task} on Hugging Face Jobs with the `hf` CLI: {pointer}, write to {bucket} ({token}), "
         "test a few rows first, and report the Job ids{report}.{ask}")


def render_short(repo: str, tag: str | None, tags: list[str], verb: str | None, gated: bool = False, dataset: str | None = None, bucket: str | None = None) -> str:
    """The one-sentence prompt for a model.

    With no dataset or bucket the prompt tells the agent to ask for them. In tests, a prompt with
    `<dataset>` and `<bucket>` placeholders led agents to leave the placeholders in their commands
    or to invent a bucket name.
    """
    kind = kind_of(tag, repo, tags)
    chosen = verb or ("train" if kind in TRAIN_BY_DEFAULT else "infer")
    data = dataset or "my dataset"
    missing = [name for name, value in (("dataset", dataset), ("bucket", bucket)) if not value]
    return SHORT.format(
        task=f"Fine-tune {repo} on {data}" if chosen == "train" else f"Run {repo} over {data}",
        pointer=SHORT_POINTERS.get((chosen, kind), SHORT_POINTERS[(chosen, "other")]),
        bucket=f"the bucket mounted with `-v hf://buckets/{bucket.removeprefix('hf://buckets/')}:/out`" if bucket else "my bucket, mounted into the Job with `-v`",
        token="this model is gated: pass `--secrets HF_TOKEN`" if gated else "a token only for gated or private repos",
        report=", cost and eval result" if chosen == "train" else " and cost per 1,000 items",
        ask=f" Ask me for the {' and the '.join(missing)} before you start." if missing else "",
    )


def mapping_table() -> str:
    lines = ["Hub task tag -> kind of model", ""]
    lines += [f"  {tag:<30} {kind}" for tag, kind in TASK_TAG_TO_KIND.items()]
    lines += [f"  {'(an OCR tag or name, see below)':<30} ocr", f"  {'(any other tag, or no tag)':<30} other", ""]
    lines += [f"OCR tags: {', '.join(sorted(OCR_TAGS))}; or `ocr` / `docling` in the model id or a tag.", ""]
    lines += ["Docs index per library: " + ", ".join(LIBRARY_DOCS), ""]
    lines += [f"Default verb: train for {', '.join(sorted(TRAIN_BY_DEFAULT))}; infer for every other kind.", "", "(verb, kind) -> pointer", ""]
    lines += [f"  {verb:<6} {kind:<11} {pointer.split('. ')[0].rstrip('.')}." for (verb, kind), (_, pointer, _) in POINTERS.items()]
    return "\n".join(lines)
