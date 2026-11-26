"""The mapping: Hub task tag -> Jobs workload -> prompt.

Everything the command prints comes from this file. To change what an agent is told for a kind
of model, edit one entry here. Hints name existing `hf` commands, so the agent reads live data
with the CLI it already has.
"""

# Hub task tag (pipeline_tag) of a model -> the workload most people want on Jobs. First match wins.
TASK_TAG_TO_WORKLOAD = {
    "feature-extraction": "embed",
    "sentence-similarity": "embed",
    "text-ranking": "embed",
    "fill-mask": "train-encoder",
    "text-classification": "classify",
    "token-classification": "classify",
    "zero-shot-classification": "classify",
    "image-classification": "classify",
    "object-detection": "classify",
    "text-generation": "generate",
    "image-text-to-text": "generate",
    "any-to-any": "generate",
    "automatic-speech-recognition": "transcribe",
}
DEFAULT_WORKLOAD = "generate"

# Workload -> what to tell the agent.
#   title:   what the run does
#   pointer: the one thing that most changes cost or correctness for this workload
#   scripts: folder of ready uv scripts to read before writing a new one
#   unit:    what "cost per 1,000" counts; None for training
#   other:   the other workload people ask for with this kind of model
WORKLOADS = {
    "generate": {
        "title": "bulk generation, classification or extraction with an LLM or VLM",
        "pointer": "Start the model's server (vLLM or SGLang, whichever the model card or the catalogue names) inside the Job and send many requests at once to localhost. A loop over `model.generate` is several times slower.",
        "scripts": "uv-scripts/vllm", "unit": "items", "other": "train-llm"},
    "ocr": {
        "title": "pages to text with an OCR model",
        "pointer": "Start the model's server (vLLM or SGLang, whichever the model card names) inside the Job and send one request per page, many at once. A loop over pages is several times slower.",
        "scripts": "uv-scripts/ocr", "unit": "pages", "other": "train-llm"},
    "embed": {
        "title": "embed a text column",
        "pointer": "Use Sentence Transformers with a large batch size and the prompt names from the model card, or Text Embeddings Inference if the catalogue lists the `tei` engine.",
        "scripts": "uv-scripts/embeddings", "unit": "texts", "other": "train-encoder"},
    "classify": {
        "title": "batch inference with a classifier",
        "pointer": "Use the library and the loading code from the model card, with batches on one small GPU.",
        "scripts": "uv-scripts/classification", "unit": "items", "other": "train-encoder"},
    "transcribe": {
        "title": "transcribe audio files",
        "pointer": "Use the library from the model card. Sort or bucket the files by length before you batch them.",
        "scripts": "uv-scripts/transcription", "unit": "audio hours", "other": "train-llm"},
    "train-llm": {
        "title": "fine-tune an LLM",
        "pointer": "Use TRL `SFTTrainer` (https://huggingface.co/docs/trl/sft_trainer). The Job disk is deleted at the end: save checkpoints and the final model to a mounted bucket or a private repo.",
        "scripts": "uv-scripts/training", "unit": None, "other": "generate"},
    "train-encoder": {
        "title": "fine-tune an encoder or an embedding model",
        "pointer": "Use the transformers `Trainer` for a classifier, or Sentence Transformers for an embedding model (https://sbert.net/docs/sentence_transformer/training_overview.html). The Job disk is deleted at the end: save the model to a mounted bucket or a private repo.",
        "scripts": "uv-scripts/classification", "unit": None, "other": "embed"},
}

PROMPT = """# {repo} on Hugging Face Jobs: {title}

Kind of model: task tag `{tag}`, library `{library}`. Workload: `{workload}`{how_chosen}.
Other workloads: `hf jobplan {repo} --task <{others}>`. The whole mapping: `hf jobplan --list`.
{custom_code}
## Find out first (each line is an existing command)
    hf models card {repo} --text
        how the authors run it: library or server, container image, flags
    hf models ls {repo}
        weight files and sizes: the GPU memory you need
    hf endpoints catalog ls --search {name}
        servers tested for this model, if it is listed
    hf datasets ls {scripts}
        ready uv scripts for this workload: read one before you write one
    hf jobs hardware
        flavors and prices
    hf jobs uv run --help
        how to launch; `hf jobs run --help` if the model needs its own image

## For this workload
{pointer}

## Then
1. Write one uv script with inline dependencies. Launch with `hf jobs uv run --detach --timeout <limit> --flavor <flavor> <script>`.
2. Run a small test first ({test}). Read the result with `hf jobs logs <id>` and open the outputs: an exit code of 0 is not a check.
3. Say which flavor you picked and why. If `hf jobs inspect <id>` still shows SCHEDULING after 10 minutes, `hf jobs cancel <id>` and use another flavor.
4. Run the full input. Write to the place the user named: mount a bucket with `-v` (see the help), or see `hf buckets --help`.
5. Report from `hf jobs inspect <id>`: Job ids, flavor, billed time, cost, {report}, and what you did not check.
"""

CUSTOM_CODE = "\nThis repo runs its own code (`custom_code`). Do not pass a token or a secret to a Job that loads it.\n"


def workload_for(tag: str | None, repo: str, tags: list[str]) -> str:
    """The default workload for a model: OCR models by name or tag, the rest by task tag."""
    if "ocr" in " ".join([repo, *tags]).lower():
        return "ocr"
    return TASK_TAG_TO_WORKLOAD.get(tag or "", DEFAULT_WORKLOAD)


def render(repo: str, tag: str | None, library: str | None, tags: list[str], task: str | None) -> str:
    workload = task or workload_for(tag, repo, tags)
    entry = WORKLOADS[workload]
    training = entry["unit"] is None
    return PROMPT.format(
        repo=repo, name=repo.split("/")[-1], tag=tag, library=library, workload=workload, title=entry["title"],
        how_chosen="" if task else " (the default for this task tag)",
        others=", ".join(w for w in WORKLOADS if w != workload),
        custom_code=CUSTOM_CODE if "custom_code" in tags else "",
        scripts=entry["scripts"], pointer=entry["pointer"],
        test="a few steps on a few rows" if training else "a few rows",
        report="the evaluation result" if training else f"cost per 1,000 {entry['unit']}",
    )


def mapping_table() -> str:
    lines = ["Hub task tag -> default workload", ""]
    lines += [f"  {tag:<30} {workload}" for tag, workload in TASK_TAG_TO_WORKLOAD.items()]
    lines += [f"  {'(a name or tag with `ocr`)':<30} ocr", f"  {'(any other tag)':<30} {DEFAULT_WORKLOAD}", "", "Workloads", ""]
    lines += [f"  {name:<14} {entry['title']}; scripts: {entry['scripts']}" for name, entry in WORKLOADS.items()]
    return "\n".join(lines)
