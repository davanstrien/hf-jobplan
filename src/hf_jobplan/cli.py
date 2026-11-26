"""hf jobplan <model>: print the prompt for running or fine-tuning a model on Hugging Face Jobs.

The command launches nothing. It makes one Hub API call to read the model's task tag, library
and tags, picks an entry from the mapping in `prompts.py`, and prints a short prompt whose
hints are existing `hf` commands. A model of an unknown kind, or one the lookup cannot read,
gets the fallback prompt.
"""
import argparse
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from hf_jobplan.prompts import mapping_table, render, render_short


def main() -> None:
    parser = argparse.ArgumentParser(prog="hf jobplan", description=__doc__.split("\n\n")[0])
    parser.add_argument("model", nargs="?", help="model id, for example google/embeddinggemma-2")
    parser.add_argument("--task", choices=["infer", "train"], help="the verb; the default comes from the kind of model")
    parser.add_argument("--short", action="store_true", help="print the one-sentence prompt, for a model page or to paste into an agent")
    parser.add_argument("--list", action="store_true", help="print the mapping and exit")
    args = parser.parse_args()
    if args.list:
        print(mapping_table())
        return
    if not args.model:
        parser.error("give a model id, or --list")
    try:
        info = HfApi().model_info(args.model, expand=["pipeline_tag", "library_name", "tags", "gated"])
    except HfHubHTTPError:
        print(render_short(args.model, None, [], args.task) if args.short else render(args.model, None, None, [], args.task, found=False))
        return
    if args.short:
        print(render_short(args.model, info.pipeline_tag, info.tags or [], args.task, gated=bool(info.gated)))
        return
    print(render(args.model, info.pipeline_tag, info.library_name, info.tags or [], args.task, gated=bool(info.gated)))


if __name__ == "__main__":
    sys.exit(main())
