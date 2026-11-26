"""Offline tests: the mapping and the rendered prompt need no network."""
import re
import subprocess

import pytest

from hf_jobplan.prompts import TASK_TAG_TO_WORKLOAD, WORKLOADS, mapping_table, render, workload_for


def test_default_workload_by_task_tag():
    assert workload_for("feature-extraction", "google/embeddinggemma-2", []) == "embed"
    assert workload_for("fill-mask", "answerdotai/ModernBERT-base", []) == "train-encoder"
    assert workload_for("image-text-to-text", "lightonai/LightOnOCR-3-0.8B", []) == "ocr"
    assert workload_for("image-text-to-text", "thinkingmachines/Inkling-2-Mini", []) == "generate"
    assert workload_for("some-new-tag", "org/model", []) == "generate"


def test_every_mapped_workload_exists():
    assert set(TASK_TAG_TO_WORKLOAD.values()) <= set(WORKLOADS)
    assert all(entry["other"] in WORKLOADS for entry in WORKLOADS.values())


def test_training_prompt_reports_evaluation_and_inference_reports_unit_cost():
    train = render("answerdotai/ModernBERT-base", "fill-mask", "transformers", [], None)
    assert "the evaluation result" in train and "cost per 1,000" not in train
    embed = render("google/embeddinggemma-2", "feature-extraction", "sentence-transformers", [], None)
    assert "cost per 1,000 texts" in embed and "(the default for this task tag)" in embed


def test_custom_code_warning_only_when_tagged():
    assert "Do not pass a token" in render("org/model", "text-generation", "transformers", ["custom_code"], "train-llm")
    assert "Do not pass a token" not in render("org/model", "text-generation", "transformers", [], "train-llm")


def test_mapping_table_lists_every_workload():
    table = mapping_table()
    assert all(name in table for name in WORKLOADS)


@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_hinted_hf_commands_exist(workload):
    """Each `hf ...` hint in the prompt names a real command: its --help must exit 0."""
    prompt = render("org/model", "text-generation", "transformers", [], workload)
    commands = set(re.findall(r"`?(hf (?:models|datasets|endpoints|jobs|buckets)(?: (?:card|ls|catalog ls|hardware|uv run|run|logs|inspect|cancel))?)", prompt))
    assert len(commands) >= 8
    for command in commands:
        done = subprocess.run([*command.split(), "--help"], check=False, capture_output=True, text=True)
        assert done.returncode == 0, command
