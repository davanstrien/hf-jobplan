"""Offline tests: the mapping and the rendered prompt need no network."""
import re
import subprocess

import pytest

from hf_jobplan.prompts import JOBS_DOCS, LIBRARY_DOCS, POINTERS, TASK_TAG_TO_KIND, kind_of, mapping_table, render

KINDS = sorted({*TASK_TAG_TO_KIND.values(), "ocr", "other"})
URLS = sorted({url.rstrip(".,;)") for _, (_, pointer, _) in POINTERS.items() for url in re.findall(r"https://[^\s`]+", pointer)} | set(LIBRARY_DOCS.values()) | {JOBS_DOCS})


def test_kind_by_task_tag_with_fallback():
    assert kind_of("feature-extraction", "google/embeddinggemma-2", []) == "embedding"
    assert kind_of("fill-mask", "answerdotai/ModernBERT-base", []) == "encoder"
    assert kind_of("image-text-to-text", "lightonai/LightOnOCR-3-0.8B", []) == "ocr"
    assert kind_of("tabular-classification", "org/model", []) == "other"
    assert kind_of(None, "org/model", []) == "other"


@pytest.mark.parametrize(("repo", "tag", "tags"), [
    ("lightonai/LightOnOCR-3-0.8B", "image-text-to-text", ["document-understanding", "tables"]),
    ("zai-org/GLM-OCR", "image-to-text", ["glm_ocr"]),
    ("PaddlePaddle/PaddleOCR-VL", "image-text-to-text", ["document-parse", "layout"]),
    ("nanonets/Nanonets-OCR2-3B", "image-text-to-text", ["pdf2markdown"]),
    ("llamaindex/OCRia-nano-1", "image-to-text", ["document-parsing"]),
    ("ds4sd/SmolDocling-256M-preview", "image-text-to-text", ["idefics3"]),
    ("stepfun-ai/GOT-OCR-2.0-hf", "image-text-to-text", ["ocr2.0"]),
])
def test_ocr_models_found_by_tag_or_name(repo, tag, tags):
    """Tags copied from the Hub on 2026-11-26."""
    assert kind_of(tag, repo, tags) == "ocr"


def test_general_vlm_is_not_ocr():
    assert kind_of("image-text-to-text", "Qwen/Qwen3.8-27B", ["qwen3_5", "eval-results"]) == "llm"
    # an OCR model with no OCR tag and no OCR word in its id gets the `llm` pointer, which gives the same server advice
    assert kind_of("image-text-to-text", "org/page-reader", ["qwen3_vl"]) == "llm"


def test_both_verbs_have_a_fallback():
    assert ("infer", "other") in POINTERS and ("train", "other") in POINTERS


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("verb", ["infer", "train", None])
def test_every_kind_and_verb_renders(kind, verb):
    tag = next((t for t, k in TASK_TAG_TO_KIND.items() if k == kind), "tabular-classification")
    repo = "org/some-ocr" if kind == "ocr" else "org/model"
    prompt = render(repo, tag, "transformers", [], verb)
    assert "hf jobs uv run --detach" in prompt and "{" not in prompt and "}" not in prompt


def test_default_verb_is_train_for_an_encoder_and_infer_for_the_rest():
    assert "Verb: `train` (the default for this kind)" in render("answerdotai/ModernBERT-base", "fill-mask", "transformers", [], None)
    assert "Verb: `infer` (the default for this kind)" in render("google/embeddinggemma-2", "feature-extraction", "sentence-transformers", [], None)


def test_training_names_the_library():
    llm = render("org/model", "text-generation", "transformers", [], "train")
    assert "trl/scripts/sft.py --model_name_or_path org/model" in llm and "docs/trl/jobs_training" in llm
    assert "transformers `Trainer`" in render("org/model", "fill-mask", "transformers", [], "train")
    assert "Sentence Transformers" in render("org/model", "sentence-similarity", "sentence-transformers", [], "train")
    assert "the evaluation result" in render("org/model", "text-generation", "transformers", [], "train")


def test_lookup_failure_prints_the_fallback():
    prompt = render("org/private-model", None, None, [], None, found=False)
    assert "lookup for this model failed" in prompt and "hf jobs uv run --detach" in prompt


def test_custom_code_warning_only_when_tagged():
    assert "Do not pass a token" in render("org/model", "text-generation", "transformers", ["custom_code"], "train")
    assert "Do not pass a token" not in render("org/model", "text-generation", "transformers", [], "train")


def test_prompt_names_the_cli():
    assert "uses the `hf` CLI" in render("org/model", None, None, [], None)


def test_mapping_table_lists_every_pointer():
    assert mapping_table().count("\n  infer") + mapping_table().count("\n  train") == len(POINTERS)


def test_hinted_hf_commands_exist():
    """Each `hf ...` hint in the prompt names a real command: its --help must exit 0."""
    prompt = render("org/model", "text-to-image", "diffusers", [], None)
    commands = set(re.findall(r"(hf (?:models card|models info|endpoints catalog ls|datasets ls|jobs hardware|jobs uv run|jobs run|jobs logs|jobs inspect|jobs cancel|buckets))", prompt))
    assert len(commands) == 11
    for command in commands:
        done = subprocess.run([*command.split(), "--help"], check=False, capture_output=True, text=True)
        assert done.returncode == 0, command


def test_diffusion_models_get_diffusers():
    assert "diffusers pipeline" in render("black-forest-labs/FLUX.1-dev", "text-to-image", "diffusers", [], None)
    assert "docs/diffusers/training/lora.md" in render("black-forest-labs/FLUX.1-dev", "text-to-image", "diffusers", [], "train")


def test_docs_line_follows_the_library():
    assert "`diffusers` index https://huggingface.co/docs/diffusers/llms.txt" in render("org/model", "text-to-image", "diffusers", [], None)
    assert "index" not in render("org/model", "tabular-classification", "some-new-lib", [], None).split("Docs as markdown")[1].split("\n")[0]


@pytest.mark.parametrize("url", URLS)
def test_linked_docs_exist(url):
    """Network test. The Hub answers 200 with an HTML page for a docs page that does not exist, so check the body too."""
    done = subprocess.run(["curl", "-sL", "-w", "\n%{http_code}", url.replace("{repo}", "org/model")], check=False, capture_output=True, text=True)
    body, _, status = done.stdout.rpartition("\n")
    assert status == "200", url
    if url.endswith((".md", "llms.txt")):
        assert "<!doctype html" not in body[:300].lower(), url
