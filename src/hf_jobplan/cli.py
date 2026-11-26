"""hf jobplan <model>: print the prompt for running a model on Hugging Face Jobs.

The command launches nothing. It makes one Hub API call to read the model's task tag, library
and tags, picks a workload from the mapping in `prompts.py`, and prints a short prompt whose
hints are existing `hf` commands.
"""
import argparse
import sys

from huggingface_hub import HfApi

from hf_jobplan.prompts import WORKLOADS, mapping_table, render


def main() -> None:
    parser = argparse.ArgumentParser(prog="hf jobplan", description=__doc__.split("\n\n")[0])
    parser.add_argument("model", nargs="?", help="model id, for example google/embeddinggemma-2")
    parser.add_argument("--task", choices=list(WORKLOADS), help="the workload; the default comes from the model's task tag")
    parser.add_argument("--list", action="store_true", help="print the mapping from task tag to workload and exit")
    args = parser.parse_args()
    if args.list:
        print(mapping_table())
        return
    if not args.model:
        parser.error("give a model id, or --list")
    info = HfApi().model_info(args.model, expand=["pipeline_tag", "library_name", "tags"])
    print(render(args.model, info.pipeline_tag, info.library_name, info.tags or [], args.task))


if __name__ == "__main__":
    sys.exit(main())
