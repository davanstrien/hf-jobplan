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
}

# OCR and document-parsing models share task tags with general VLMs, so the task tag cannot find them.
# A model is kind `ocr` when one of its tags is in OCR_TAGS, or when its id or a tag contains one of OCR_WORDS.
OCR_TAGS = {"ocr", "document-parse", "document-parsing", "document-understanding", "pdf2markdown", "document-ai", "handwriting-recognition"}
OCR_WORDS = ("ocr", "docling")

# Kinds that people mostly fine-tune. Every other kind defaults to `infer`.
TRAIN_BY_DEFAULT = {"encoder"}

SAVE = "The Job disk is deleted at the end: save checkpoints and the final model to a mounted bucket or a private repo."
SERVER = "Start the model's server (vLLM or SGLang, whichever the model card or the catalogue names) inside the Job and send many requests at once to localhost. A loop over `model.generate` is several times slower."

# (verb, kind) -> (what the run is, the one pointer that most changes cost or correctness, folder of ready uv scripts).
# (verb, "other") is the fallback for every kind without its own entry.
POINTERS = {
    ("infer", "llm"): ("bulk generation, classification or extraction", SERVER, "uv-scripts/vllm"),
    ("infer", "ocr"): ("pages to text", "Follow the model card. If it gives a vLLM or SGLang command, start that server inside the Job and send one request per page to localhost, many at once: a loop over pages is several times slower. If the model has its own library (PaddleOCR, ONNX, ...), use that library with batches.", "uv-scripts/ocr"),
    ("infer", "embedding"): ("embed a dataset", "Use Sentence Transformers with a large batch size and the prompt names from the model card, or Text Embeddings Inference if the catalogue lists the `tei` engine.", "uv-scripts/embeddings"),
    ("infer", "classifier"): ("batch inference", "Use the transformers `pipeline` or the loading code from the model card, with batches on one small GPU.", "uv-scripts/classification"),
    ("infer", "speech"): ("transcribe audio", "Use the library from the model card. Sort the files by length before you batch them.", "uv-scripts/transcription"),
    ("infer", "other"): ("batch inference", "Use the library and the loading code from the model card. If the card has none, the library name above and the files in the repo show how to load the model.", None),
    ("train", "llm"): ("fine-tune", f"Use TRL `SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer). {SAVE}", "uv-scripts/training"),
    ("train", "ocr"): ("fine-tune", f"Use TRL `SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer) or the fine-tuning code the model card links. {SAVE}", "uv-scripts/transformers-training"),
    ("train", "embedding"): ("fine-tune", f"Use Sentence Transformers (https://sbert.net/docs/sentence_transformer/training_overview.html) and run an evaluator before and after. {SAVE}", "uv-scripts/embeddings"),
    ("train", "encoder"): ("fine-tune", f"Use the transformers `Trainer` (https://huggingface.co/docs/transformers/training) for a classifier, or Sentence Transformers (https://sbert.net/docs/sentence_transformer/training_overview.html) for an embedding model. {SAVE}", "uv-scripts/classification"),
    ("train", "classifier"): ("fine-tune", f"Use the transformers `Trainer` (https://huggingface.co/docs/transformers/training). {SAVE}", "uv-scripts/classification"),
    ("train", "other"): ("fine-tune", f"Use the training method that the model card or its library documents. The transformers `Trainer` (https://huggingface.co/docs/transformers/training) covers most transformers models. {SAVE}", None),
}

PROMPT = """# {verb_title} {repo} on Hugging Face Jobs{title}

Everything below uses the `hf` CLI, which printed this. `hf --help` lists its commands, and every command takes `--help`.

{facts}
The other verb: `hf jobplan {repo} --task {other_verb}`. The whole mapping: `hf jobplan --list`.
{custom_code}
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

## For this kind of model
{pointer}

## Then
1. Write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first ({test}). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write to the place the user named: mount a bucket with `-v` (see the help), or see `hf buckets --help`.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, {report}, and what you did not check.
"""

CUSTOM_CODE = "\nThis repo runs its own code (`custom_code`). Do not pass a token or a secret to a Job that loads it.\n"


def kind_of(tag: str | None, repo: str, tags: list[str]) -> str:
    """OCR models by tag or name, the rest by task tag, `other` when the tag is unknown or missing."""
    lowered = [t.lower() for t in tags]
    if OCR_TAGS & set(lowered) or any(word in text for word in OCR_WORDS for text in [repo.lower(), *lowered]):
        return "ocr"
    return TASK_TAG_TO_KIND.get(tag or "", "other")


def render(repo: str, tag: str | None, library: str | None, tags: list[str], verb: str | None, found: bool = True) -> str:
    """The prompt for one model. `found=False` means the Hub lookup failed: print the fallback."""
    kind = kind_of(tag, repo, tags)
    chosen = verb or ("train" if kind in TRAIN_BY_DEFAULT else "infer")
    title, pointer, scripts = POINTERS.get((chosen, kind), POINTERS[(chosen, "other")])
    if not found:
        facts = "The Hub lookup for this model failed (private, gated, or a wrong id), so the kind of model is unknown. The steps below still apply."
    else:
        why = "" if verb else " (the default for this kind)"
        facts = f"Task tag `{tag}`, library `{library}` -> kind `{kind}`. Verb: `{chosen}`{why}."
    return PROMPT.format(
        repo=repo, name=repo.split("/")[-1], verb_title="Fine-tune" if chosen == "train" else "Run", title="" if chosen == "train" else f": {title}", facts=facts,
        other_verb="infer" if chosen == "train" else "train", custom_code=CUSTOM_CODE if "custom_code" in tags else "",
        scripts=f"hf datasets ls {scripts}" if scripts else "hf datasets ls --author uv-scripts", pointer=pointer,
        test="a few steps on a few rows" if chosen == "train" else "a few rows",
        report="the evaluation result" if chosen == "train" else "cost per 1,000 items",
    )


def mapping_table() -> str:
    lines = ["Hub task tag -> kind of model", ""]
    lines += [f"  {tag:<30} {kind}" for tag, kind in TASK_TAG_TO_KIND.items()]
    lines += [f"  {'(an OCR tag or name, see below)':<30} ocr", f"  {'(any other tag, or no tag)':<30} other", ""]
    lines += [f"OCR tags: {', '.join(sorted(OCR_TAGS))}; or `ocr` / `docling` in the model id or a tag.", ""]
    lines += [f"Default verb: train for {', '.join(sorted(TRAIN_BY_DEFAULT))}; infer for every other kind.", "", "(verb, kind) -> pointer", ""]
    lines += [f"  {verb:<6} {kind:<11} {pointer.split('. ')[0].rstrip('.')}." for (verb, kind), (_, pointer, _) in POINTERS.items()]
    return "\n".join(lines)
