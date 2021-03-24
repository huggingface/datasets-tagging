import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import langcodes as lc
import streamlit as st
import yaml
from datasets.utils.metadata import (DatasetMetadata, known_creators,
                                     known_licenses, known_multilingualities,
                                     known_size_categories, known_task_ids)

st.set_page_config(
    page_title="HF Dataset Tagging App",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# XXX: restyling errors as streamlit does not respect whitespaces on `st.error` and doesn't scroll horizontally, which
#   generally makes things easier when reading error reports
st.markdown(
    """
<style>
    div[role=alert] { overflow-x: scroll}
    div.stAlert p { white-space: pre }
</style>
""",
    unsafe_allow_html=True,
)

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
    w.markdown(f"#### {title}")
    if len(invalid_values) > 0:
        w.markdown("Found the following invalid values:")
        w.error(invalid_values)
    return w.multiselect(markdown, valid_set, default=valid_values, format_func=format_func)


def validate_dict(w: st.delta_generator.DeltaGenerator, state_dict: Dict):
    try:
        DatasetMetadata(**state_dict)
        w.markdown("✅ This is a valid tagset! 🤗")
    except Exception as e:
        w.markdown("❌ This is an invalid tagset, here are the errors in it:")
        w.error(e)


def new_state() -> Dict[str, List]:
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


def is_state_empty(state: Dict[str, List]) -> bool:
    return sum(len(v) if v is not None else 0 for v in state.values()) == 0


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
if not is_state_empty(state):
    if rightbtn.button("flush state"):
        state = new_state()
        initial_state = None
        preloaded_id = None
        st.experimental_set_query_params()

if preloaded_id is not None and initial_state is not None:
    st.sidebar.markdown(
        f"""
---
The current base tagset is [`{preloaded_id}`](https://huggingface.co/datasets/{preloaded_id})
"""
    )
    validate_dict(st.sidebar, initial_state)
    st.sidebar.markdown(
        f"""
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
    valid_set=list(known_task_ids.keys()),
    format_func=lambda tg: f"{tg}: {known_task_ids[tg]['description']}",
)
task_specifics = []
for tg in state["task_categories"]:
    specs = multiselect(
        leftcol,
        f"Specific _{tg}_ tasks",
        f"What specific tasks does the dataset support?",
        values=[ts for ts in (state["task_ids"] or []) if ts in known_task_ids[tg]["options"]],
        valid_set=known_task_ids[tg]["options"],
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
    valid_set=list(known_multilingualities.keys()),
    format_func=lambda m: f"{m} : {known_multilingualities[m]}",
)

if "other" in state["multilinguality"]:
    other_multilinguality = st.text_input(
        "You selected 'other' type of multilinguality. Please enter a short hyphen-separated description:",
        value="my-multilinguality",
    )
    st.write(f"Registering other-{other_multilinguality} multilinguality")
    state["multilinguality"][state["multilinguality"].index("other")] = f"other-{other_multilinguality}"

valid_values, invalid_values = list(), list()
for langtag in state["languages"]:
    try:
        lc.get(langtag)
        valid_values.append(langtag)
    except:
        invalid_values.append(langtag)
leftcol.markdown("#### Languages")
if len(invalid_values) > 0:
    leftcol.markdown("Found the following invalid values:")
    leftcol.error(invalid_values)

langtags = leftcol.text_area(
    "What languages are represented in the dataset? expected format is BCP47 tags separated for ';' e.g. 'en-US;fr-FR'",
    value=";".join(valid_values),
)
state["languages"] = langtags.split(";")

leftcol.markdown("### Dataset creators")
state["language_creators"] = multiselect(
    leftcol,
    "Data origin",
    "Where does the text in the dataset come from?",
    values=state["language_creators"],
    valid_set=known_creators["language"],
)
state["annotations_creators"] = multiselect(
    leftcol,
    "Annotations origin",
    "Where do the annotations in the dataset come from?",
    values=state["annotations_creators"],
    valid_set=known_creators["annotations"],
)


state["licenses"] = multiselect(
    leftcol,
    "Licenses",
    "What licenses is the dataset under?",
    valid_set=list(known_licenses.keys()),
    values=state["licenses"],
    format_func=lambda l: f"{l} : {known_licenses[l]}",
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

current_size_cats = state.get("size_categories") or ["unknown"]
ok, nonok = split_known(current_size_cats, known_size_categories)
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
state["size_categories"] = [
    leftcol.selectbox(
        "What is the size category of the dataset?",
        options=known_size_categories,
        index=known_size_categories.index(ok[0]) if len(ok) > 0 else 0,
    )
]


########################
## Show results
########################

rightcol.markdown(
    f"""
### Finalized tag set

"""
)
if is_state_empty(state):
    rightcol.markdown("❌ This is an invalid tagset: it's empty!")
else:
    validate_dict(rightcol, state)


rightcol.markdown(
    f"""

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
    validate_dict(rightcol, inputdict)
