---
title: Datasets Tagging
emoji: 🤗
colorFrom: pink
colorTo: blue
sdk: streamlit
app_file: tagging_app.py
pinned: false
---

# datasets-tagging
A Streamlit app to add structured tags to the datasets - available on line [here!](https://huggingface.co/datasets/tagging)


1. `pip install -r requirements.txt`
2. `./build_metadata_file.py` will build an up-to-date metadata file from the `datasets/` repo (clones it locally)
3. `streamlit run tagging_app.py`

This will give you a `localhost` link you can click to open in your browser.

The app initialization on the first run takes a few minutes, subsequent runs are faster.

Make sure to hit the `Done? Save to File!` button in the right column when you're done tagging a config!

