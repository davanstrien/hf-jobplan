"""Offline tests for the parts of `hf jobplan` that need no network."""
from types import SimpleNamespace

from hf_jobplan.cli import card_run_lines, default_task

CARD = """# Model

```bash
docker run --gpus all \\
  lmsysorg/sglang:dev-clef \\
  sglang serve --model-path org/model
```

```python
print("hello")
```

```bash
pip install -U transformers
```
"""


def test_card_blocks_keep_whole_serve_command():
    blocks = card_run_lines(CARD)
    assert len(blocks) == 2
    assert "lmsysorg/sglang:dev-clef" in blocks[0] and "sglang serve" in blocks[0]
    assert blocks[1] == "pip install -U transformers"


def test_card_without_run_blocks():
    assert card_run_lines("# Model\n\nNo code here.\n") == []


def info(model_id, tag, tags=()):
    return SimpleNamespace(id=model_id, pipeline_tag=tag, tags=list(tags))


def test_default_task_by_model_kind():
    assert default_task(info("google/embeddinggemma-2", "feature-extraction")) == "embed"
    assert default_task(info("answerdotai/ModernBERT-base", "fill-mask")) == "train-encoder"
    assert default_task(info("lightonai/LightOnOCR-3-0.8B", "image-text-to-text")) == "ocr"
    assert default_task(info("thinkingmachines/Inkling-2-Mini", "image-text-to-text")) == "generate"
