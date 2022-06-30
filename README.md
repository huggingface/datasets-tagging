---
title: Datasets Tagging
emoji: 🤗
colorFrom: pink
colorTo: blue
sdk: streamlit
app_file: tagging_app.py
pinned: false
---

## ⚠️ This repo is now directly maintained in the Space repo at https://huggingface.co/spaces/huggingface/datasets-tagging ⚠️

You can clone it from there with `git clone https://huggingface.co/spaces/huggingface/datasets-tagging`.

You can open Pull requests & Discussions in the repo too: https://huggingface.co/spaces/huggingface/datasets-tagging/discussions.


# 🤗 Datasets Tagging
A Streamlit app to add structured tags to a dataset card.
Available online [here!](https://huggingface.co/spaces/huggingface/datasets-tagging)


1. `pip install -r requirements.txt`
2. `./build_metadata_file.py` will build an up-to-date metadata file from the `datasets/` repo (clones it locally)
3. `streamlit run tagging_app.py`

This will give you a `localhost` link you can click to open in your browser.

The app initialization on the first run takes a few minutes, subsequent runs are faster.

Make sure to hit the `Done? Save to File!` button in the right column when you're done tagging a config!
