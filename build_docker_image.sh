#!/usr/bin/env bash


cleanup() {
  rm -f Dockerfile
}

trap cleanup ERR EXIT

cat > Dockerfile << EOF
FROM python
COPY requirements.txt .
COPY tagging_app.py .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "tagging_app.py"]
EOF

set -eEx

./build_metadata_file.py
docker build -t dataset-tagger .
