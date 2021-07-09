from typing import Dict, List


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
        "pretty_name": None,
    }
