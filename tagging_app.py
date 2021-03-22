import json
from pathlib import Path
from typing import List, Tuple

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


@st.cache(allow_output_mutation=True)
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
    return [v for v in vals if v in okset], [v for v in vals if v not in okset]


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


qp = st.experimental_get_query_params()
preload = qp.get("preload_dataset", list())
preloaded_id = None
did_index = 0
if len(preload) == 1 and preload[0] in all_dataset_ids:
    preloaded_id, *_ = preload
    pre_loaded = existing_tag_sets[preloaded_id] or new_pre_loaded()
    did_index = all_dataset_ids.index(preloaded_id)

did = st.sidebar.selectbox(label="Choose dataset to load tag set from", options=all_dataset_ids, index=did_index)

leftbtn, rightbtn = st.sidebar.beta_columns(2)
if leftbtn.button("pre-load tagset"):
    pre_loaded = existing_tag_sets[did] or new_pre_loaded()
    st.experimental_set_query_params(preload_dataset=did)
if rightbtn.button("flush state"):
    pre_loaded = new_pre_loaded()
    st.experimental_set_query_params()

if preloaded_id is not None:
    st.sidebar.markdown(f"Took [{preloaded_id}](https://huggingface.co/datasets/{preloaded_id}) as base tagset.")


leftcol, _, rightcol = st.beta_columns([12, 1, 12])


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
filtered_existing_languages = [lgc for lgc in set(pre_loaded["languages"]) if lgc not in language_set_restricted]
pre_loaded["languages"] = [lgc for lgc in set(pre_loaded["languages"]) if lgc in language_set_restricted]

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

if len(filtered_existing_languages) > 0:
    leftcol.markdown(f"**Found bad language codes in existing tagset**:\n{filtered_existing_languages}")
languages = leftcol.multiselect(
    "What languages are represented in the dataset?",
    options=list(language_set_restricted.keys()),
    default=pre_loaded["languages"],
    format_func=lambda m: f"{m} : {language_set_restricted[m]}",
)


leftcol.markdown("### Dataset creators")
ok, nonok = split_known(pre_loaded["language_creators"], creator_set["language"])
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
language_creators = leftcol.multiselect(
    "Where does the text in the dataset come from?",
    options=creator_set["language"],
    default=ok,
)
ok, nonok = split_known(pre_loaded["annotations_creators"], creator_set["annotations"])
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
annotations_creators = leftcol.multiselect(
    "Where do the annotations in the dataset come from?",
    options=creator_set["annotations"],
    default=ok,
)

ok, nonok = split_known(pre_loaded["licenses"], list(license_set.keys()))
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
licenses = leftcol.multiselect(
    "What licenses is the dataset under?",
    options=list(license_set.keys()),
    default=ok,
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

size_cats = ["unknown", "n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "n>1M"]
current_size_cats = pre_loaded.get("size_categories") or ["unknown"]
ok, nonok = split_known(current_size_cats, size_cats)
if len(nonok) > 0:
    leftcol.markdown(f"**Found bad codes in existing tagset**:\n{nonok}")
size_category = leftcol.selectbox(
    "What is the size category of the dataset?",
    options=size_cats,
    index=size_cats.index(ok[0]) if len(ok) > 0 else 0,
)


########################
## Show results
########################
yamlblock = yaml.dump(
    {
        "task_categories": task_categories,
        "task_ids": task_specifics,
        "multilinguality": multilinguality,
        "languages": languages,
        "language_creators": language_creators,
        "annotations_creators": annotations_creators,
        "source_datasets": source_datasets,
        "size_categories": size_category,
        "licenses": licenses,
    }
)
rightcol.markdown(
    f"""
### Finalized tag set

Copy it into your dataset's `README.md` header! 🤗 

```yaml
{yamlblock}
```""",
)
