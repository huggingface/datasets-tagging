#!/usr/bin/env python

""" This script will clone the `datasets` repository in your current directory and parse all currently available
    metadata, from the `README.md` yaml headers and the automatically generated json files.
    It dumps the results in a `metadata_{current-commit-of-datasets}.json` file.
"""

import json
from pathlib import Path
from subprocess import check_call, check_output
from typing import Dict

import yaml

from apputils import new_state


def metadata_from_readme(f: Path) -> Dict:
    with f.open() as fi:
        content = [line.strip() for line in fi]

    if content[0] == "---" and "---" in content[1:]:
        yamlblock = "\n".join(content[1 : content[1:].index("---") + 1])
        return yaml.safe_load(yamlblock) or dict()


def load_ds_datas():
    drepo = Path("datasets")
    if drepo.exists() and drepo.is_dir():
        check_call(["git", "pull"], cwd=drepo)
    else:
        check_call(["git", "clone", "https://github.com/huggingface/datasets.git"])
    head_sha = check_output(["git", "rev-parse", "HEAD"], cwd=drepo)

    datasets_md = dict()

    for ddir in sorted((drepo / "datasets").iterdir(), key=lambda d: d.name):

        try:
            metadata = metadata_from_readme(ddir / "README.md")
        except:
            metadata = None
        if metadata is None or len(metadata) == 0:
            metadata = new_state()

        try:
            with (ddir / "dataset_infos.json").open() as fi:
                infos = json.load(fi)
        except:
            infos = None

        datasets_md[ddir.name] = dict(metadata=metadata, infos=infos)
    return head_sha.decode().strip(), datasets_md


if __name__ == "__main__":
    head_sha, datas = load_ds_datas()
    fn = f"metadata_{head_sha}.json"
    print(f"writing to '{fn}'")
    with open(fn, "w") as fi:
        fi.write(json.dumps(datas))
