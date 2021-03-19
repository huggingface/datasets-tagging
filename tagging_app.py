import json
import os
from dataclasses import asdict
from glob import glob

import datasets
import streamlit as st
import yaml

st.set_page_config(
    page_title="HF Dataset Tagging App",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

task_set = json.load(open("task_set.json"))
license_set = json.load(open("license_set.json"))
language_set_restricted = json.load(open("language_set.json"))
language_set = json.load(open("language_set_full.json"))

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


def load_existing_tags():
    has_tags = {}
    for fname in glob("saved_tags/*/*/tags.json"):
        _, did, cid, _ = fname.split(os.sep)
        has_tags[did] = has_tags.get(did, {})
        has_tags[did][cid] = fname
    return has_tags


def new_pre_loaded():
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


pre_loaded = new_pre_loaded()

existing_tag_sets = load_existing_tags()
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


qp = st.experimental_get_query_params()
preload = qp.get("preload_dataset", list())
did_index = 2
if len(preload) == 1 and preload[0] in all_dataset_ids:
    did_qp, *_ = preload
    cid_qp = next(iter(existing_tag_sets[did_qp]))
    pre_loaded = json.load(open(existing_tag_sets[did_qp][cid_qp]))
    did_index = all_dataset_ids.index(did_qp)

did = st.sidebar.selectbox(label="Choose dataset to load tag set from", options=all_dataset_ids, index=did_index)
if len(existing_tag_sets[did]) > 1:
    cid = st.sidebar.selectbox(
        label="Choose config to load tag set from",
        options=list(existing_tag_sets[did].keys()),
        index=0,
    )
else:
    cid = next(iter(existing_tag_sets[did].keys()))

if st.sidebar.button("pre-load this tag set"):
    pre_loaded = json.load(open(existing_tag_sets[did][cid]))
    st.experimental_set_query_params(preload_dataset=did)
if st.sidebar.button("flush state"):
    pre_loaded = new_pre_loaded()
    st.experimental_set_query_params()

leftcol, _, rightcol = st.beta_columns([12, 1, 12])


pre_loaded["languages"] = list(set(pre_loaded["languages"]))

leftcol.markdown("### Supported tasks")
task_categories = leftcol.multiselect(
    "What categories of task does the dataset support?",
    options=list(task_set.keys()),
    default=pre_loaded["task_categories"],
    format_func=lambda tg: f"{tg} : {task_set[tg]['description']}",
)
task_specifics = []
for tg in task_categories:
    task_specs = leftcol.multiselect(
        f"What specific *{tg}* tasks does the dataset support?",
        options=task_set[tg]["options"],
        default=[ts for ts in pre_loaded["task_ids"] if ts in task_set[tg]["options"]],
    )
    if "other" in task_specs:
        other_task = st.text_input(
            "You selected 'other' task. Please enter a short hyphen-separated description for the task:",
            value="my-task-description",
        )
        st.write(f"Registering {tg}-other-{other_task} task")
        task_specs[task_specs.index("other")] = f"{tg}-other-{other_task}"
    task_specifics += task_specs

leftcol.markdown("### Languages")
multilinguality = leftcol.multiselect(
    "Does the dataset contain more than one language?",
    options=list(multilinguality_set.keys()),
    default=pre_loaded["multilinguality"],
    format_func=lambda m: f"{m} : {multilinguality_set[m]}",
)
if "other" in multilinguality:
    other_multilinguality = st.text_input(
        "You selected 'other' type of multilinguality. Please enter a short hyphen-separated description:",
        value="my-multilinguality",
    )
    st.write(f"Registering other-{other_multilinguality} multilinguality")
    multilinguality[multilinguality.index("other")] = f"other-{other_multilinguality}"
languages = leftcol.multiselect(
    "What languages are represented in the dataset?",
    options=list(language_set.keys()),
    default=pre_loaded["languages"],
    format_func=lambda m: f"{m} : {language_set[m]}",
)

leftcol.markdown("### Dataset creators")
language_creators = leftcol.multiselect(
    "Where does the text in the dataset come from?",
    options=creator_set["language"],
    default=pre_loaded["language_creators"],
)
annotations_creators = leftcol.multiselect(
    "Where do the annotations in the dataset come from?",
    options=creator_set["annotations"],
    default=pre_loaded["annotations_creators"],
)
licenses = leftcol.multiselect(
    "What licenses is the dataset under?",
    options=list(license_set.keys()),
    default=pre_loaded["licenses"],
    format_func=lambda l: f"{l} : {license_set[l]}",
)
if "other" in licenses:
    other_license = st.text_input(
        "You selected 'other' type of license. Please enter a short hyphen-separated description:",
        value="my-license",
    )
    st.write(f"Registering other-{other_license} license")
    licenses[licenses.index("other")] = f"other-{other_license}"
# link ro supported datasets
pre_select_ext_a = []
if "original" in pre_loaded["source_datasets"]:
    pre_select_ext_a += ["original"]
if any([p.startswith("extended") for p in pre_loaded["source_datasets"]]):
    pre_select_ext_a += ["extended"]
extended = leftcol.multiselect(
    "Does the dataset contain original data and/or was it extended from other datasets?",
    options=["original", "extended"],
    default=pre_select_ext_a,
)
source_datasets = ["original"] if "original" in extended else []
if "extended" in extended:
    pre_select_ext_b = [p.split("|")[1] for p in pre_loaded["source_datasets"] if p.startswith("extended")]
    extended_sources = leftcol.multiselect(
        "Which other datasets does this one use data from?",
        options=all_dataset_ids + ["other"],
        default=pre_select_ext_b,
    )
    if "other" in extended_sources:
        other_extended_sources = st.text_input(
            "You selected 'other' dataset. Please enter a short hyphen-separated description:",
            value="my-dataset",
        )
        st.write(f"Registering other-{other_extended_sources} dataset")
        extended_sources[extended_sources.index("other")] = f"other-{other_extended_sources}"
    source_datasets += [f"extended|{src}" for src in extended_sources]
size_category = leftcol.selectbox(
    "What is the size category of the dataset?",
    options=["unknown", "n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "n>1M"],
    index=["unknown", "n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "n>1M"].index(
        (pre_loaded.get("size_categories") or ["unknown"])[0]
    ),
)


########################
## Show results
########################
rightcol.markdown(
    f"""
### Finalized tag set
```yaml
{yaml.dump({
    "task_categories": task_categories,
    "task_ids": task_specifics,
    "multilinguality": multilinguality,
    "languages": languages,
    "language_creators": language_creators,
    "annotations_creators": annotations_creators,
    "source_datasets": source_datasets,
    "size_categories": size_category,
    "licenses": licenses,
})}
```
"""
)
