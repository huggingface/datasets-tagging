import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import streamlit as st
import yaml
from datasets.utils.metadata_validator import DatasetMetadata

st.set_page_config(
    page_title="HF Dataset Tagging App",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

task_set = json.load(open("task_set.json"))
license_set = json.load(open("license_set.json"))
language_set_restricted = json.load(open("language_set.json"))

multilinguality_set = {
    "monolingual": "contains a single language",
    "multilingual": "contains multiple languages",
    "translation": "contains translated or aligned text",
    "other": "other type of language distribution",
}

creator_set = {
    "language": [
        "found",
        "crowdsourced",
        "expert-generated",
        "machine-generated",
        "other",
    ],
    "annotations": [
        "found",
        "crowdsourced",
        "expert-generated",
        "machine-generated",
        "no-annotation",
        "other",
    ],
}

########################
## Helper functions
########################


def load_ds_datas():
    metada_exports = sorted(
        [f for f in Path.cwd().iterdir() if f.name.startswith("metadata_")],
        key=lambda f: f.lstat().st_mtime,
        reverse=True,
    )
    if len(metada_exports) == 0:
        raise ValueError("need to run ./build_metada_file.py at least once")
    with metada_exports[0].open() as fi:
        return json.load(fi)


def split_known(vals: List[str], okset: List[str]) -> Tuple[List[str], List[str]]:
    if vals is None:
        return [], []
    return [v for v in vals if v in okset], [v for v in vals if v not in okset]


def multiselect(
    w: st.delta_generator.DeltaGenerator,
    title: str,
    markdown: str,
    values: List[str],
    valid_set: List[str],
    format_func: Callable = str,
):
    valid_values, invalid_values = split_known(values, valid_set)
    w.markdown(
        """
#### {title}
{errors}
""".format(
            title=title, errors="" if len(invalid_values) == 0 else f"_Found invalid values:_ `{invalid_values}`"
        )
    )
    return w.multiselect(markdown, valid_set, default=valid_values, format_func=format_func)


def validate_dict(state_dict: Dict) -> str:
    try:
        DatasetMetadata(**state_dict)
        valid = "✔️ This is a valid tagset! 🤗"
    except Exception as e:
        valid = f"""
🙁 This is an invalid tagset, here are the errors in it:
```
{e}
```
You're _very_ welcome to fix these issues and submit a new PR on [`datasets`](https://github.com/huggingface/datasets/)
        """
    return valid


def new_state():
    return {
        "task_categories": [],
        "task_ids": [],
        "multilinguality": [],
        "languages": [],
        "language_creators": [],
        "annotations_creators": [],
        "source_datasets": [],
        "size_categories": [],
        "licenses": [],
    }


state = new_state()
datasets_md = load_ds_datas()
existing_tag_sets = {name: mds["metadata"] for name, mds in datasets_md.items()}
all_dataset_ids = list(existing_tag_sets.keys())


########################
## Dataset selection
########################


st.sidebar.markdown(
    """
# HuggingFace Dataset Tagger

This app aims to make it easier to add structured tags to the datasets present in the library.

Each configuration requires its own tasks, as these often correspond to distinct sub-tasks. However, we provide the opportunity
to pre-load the tag sets from another dataset or configuration to avoid too much redundancy.

The tag sets are saved in JSON format, but you can print a YAML version in the right-most column to copy-paste to the config README.md

### Preloading an existing tag set

You can load an existing tag set to get started if you want.
Beware that clicking pre-load will overwrite the current state!
"""
)


queryparams = st.experimental_get_query_params()
preload = queryparams.get("preload_dataset", list())
preloaded_id = None
initial_state = None
did_index = 0
if len(preload) == 1 and preload[0] in all_dataset_ids:
    preloaded_id, *_ = preload
    initial_state = existing_tag_sets.get(preloaded_id)
    state = initial_state or new_state()
    did_index = all_dataset_ids.index(preloaded_id)

preloaded_id = st.sidebar.selectbox(
    label="Choose dataset to load tag set from", options=all_dataset_ids, index=did_index
)
leftbtn, rightbtn = st.sidebar.beta_columns(2)
if leftbtn.button("pre-load"):
    initial_state = existing_tag_sets[preloaded_id]
    state = initial_state or new_state()
    st.experimental_set_query_params(preload_dataset=preloaded_id)
if rightbtn.button("flush state"):
    state = new_state()
    initial_state = None
    preloaded_id = None
    st.experimental_set_query_params()

if preloaded_id is not None and initial_state is not None:
    valid = validate_dict(initial_state)
    st.sidebar.markdown(
        f"""
---
The current base tagset is [`{preloaded_id}`](https://huggingface.co/datasets/{preloaded_id})
{valid}
Here is the matching yaml block:

```yaml
{yaml.dump(initial_state)}
```
"""
    )


leftcol, _, rightcol = st.beta_columns([12, 1, 12])


leftcol.markdown("### Supported tasks")
state["task_categories"] = multiselect(
    leftcol,
    "Task category",
    "What categories of task does the dataset support?",
    values=state["task_categories"],
    valid_set=list(task_set.keys()),
    format_func=lambda tg: f"{tg}: {task_set[tg]['description']}",
)
task_specifics = []
for tg in state["task_categories"]:
    specs = multiselect(
        leftcol,
        f"Specific _{tg}_ tasks",
        f"What specific tasks does the dataset support?",
        values=[ts for ts in (state["task_ids"] or []) if ts in task_set[tg]["options"]],
        valid_set=task_set[tg]["options"],
    )
    if "other" in specs:
        other_task = st.text_input(
            "You selected 'other' task. Please enter a short hyphen-separated description for the task:",
            value="my-task-description",
        )
        st.write(f"Registering {tg}-other-{other_task} task")
        specs[specs.index("other")] = f"{tg}-other-{other_task}"
    task_specifics += specs
state["task_ids"] = task_specifics


leftcol.markdown("### Languages")
state["multilinguality"] = multiselect(
    leftcol,
    "Monolingual?",
    "Does the dataset contain more than one language?",
    values=state["multilinguality"],
    valid_set=list(multilinguality_set.keys()),
    format_func=lambda m: f"{m} : {multilinguality_set[m]}",
)

if "other" in state["multilinguality"]:
    other_multilinguality = st.text_input(
        "You selected 'other' type of multilinguality. Please enter a short hyphen-separated description:",
        value="my-multilinguality",
    )
    st.write(f"Registering other-{other_multilinguality} multilinguality")
    state["multilinguality"][state["multilinguality"].index("other")] = f"other-{other_multilinguality}"

state["languages"] = multiselect(
    leftcol,
    "Languages",
    "What languages are represented in the dataset?",
    values=state["languages"],
    valid_set=list(language_set_restricted.keys()),
    format_func=lambda m: f"{m} : {language_set_restricted[m]}",
)


leftcol.markdown("### Dataset creators")
state["language_creators"] = multiselect(
    leftcol,
    "Data origin",
    "Where does the text in the dataset come from?",
    values=state["language_creators"],
    valid_set=creator_set["language"],
)
state["annotations_creators"] = multiselect(
    leftcol,
    "Annotations origin",
    "Where do the annotations in the dataset come from?",
    values=state["annotations_creators"],
    valid_set=creator_set["annotations"],
)


state["licenses"] = multiselect(
    leftcol,
    "Licenses",
    "What licenses is the dataset under?",
    valid_set=list(license_set.keys()),
    values=state["licenses"],
    format_func=lambda l: f"{l} : {license_set[l]}",
)
if "other" in state["licenses"]:
    other_license = st.text_input(
        "You selected 'other' type of license. Please enter a short hyphen-separated description:",
        value="my-license",
    )
    st.write(f"Registering other-{other_license} license")
    state["licenses"][state["licenses"].index("other")] = f"other-{other_license}"

# link to supported datasets
pre_select_ext_a = []
if "original" in state["source_datasets"]:
    pre_select_ext_a += ["original"]
if any([p.startswith("extended") for p in state["source_datasets"]]):
    pre_select_ext_a += ["extended"]
state["extended"] = multiselect(
    leftcol,
    "Relations to existing work",
    "Does the dataset contain original data and/or was it extended from other datasets?",
    values=pre_select_ext_a,
    valid_set=["original", "extended"],
)
state["source_datasets"] = ["original"] if "original" in state["extended"] else []

if "extended" in state["extended"]:
    pre_select_ext_b = [p.split("|")[1] for p in state["source_datasets"] if p.startswith("extended")]
    extended_sources = multiselect(
        leftcol,
        "Linked datasets",
        "Which other datasets does this one use data from?",
        values=pre_select_ext_b,
        valid_set=all_dataset_ids + ["other"],
    )
    if "other" in extended_sources:
        other_extended_sources = st.text_input(
            "You selected 'other' dataset. Please enter a short hyphen-separated description:",
            value="my-dataset",
        )
        st.write(f"Registering other-{other_extended_sources} dataset")
        extended_sources[extended_sources.index("other")] = f"other-{other_extended_sources}"
    state["source_datasets"] += [f"extended|{src}" for src in extended_sources]

size_cats = ["unknown", "n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "n>1M"]
current_size_cats = state.get("size_categories") or ["unknown"]
ok, nonok = split_known(current_size_cats, size_cats)
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
state["size_categories"] = [
    leftcol.selectbox(
        "What is the size category of the dataset?",
        options=size_cats,
        index=size_cats.index(ok[0]) if len(ok) > 0 else 0,
    )
]


########################
## Show results
########################

valid = validate_dict(state)
rightcol.markdown(
    f"""
### Finalized tag set

{valid}

```yaml
{yaml.dump(state)}
```
---
#### Arbitrary yaml validator

This is a standalone tool, it is useful to check for errors on an existing tagset or modifying directly the text rather than the UI on the left.
""",
)

yamlblock = rightcol.text_area("Input your yaml here")
if yamlblock.strip() != "":
    inputdict = yaml.safe_load(yamlblock)
    valid = validate_dict(inputdict)
    rightcol.markdown(valid)
