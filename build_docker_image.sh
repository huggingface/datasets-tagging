#!/usr/bin/env bash

cleanup() {
  rm -f Dockerfile .dockerignore
}

trap cleanup ERR EXIT

./build_metadata_file.py

cat > .dockerignore << EOF
.git
datasets
EOF

cat > Dockerfile << EOF
FROM python
COPY requirements.txt tagging_app.py task_set.json language_set.json license_set.json metadata_927d44346b12fac66e97176608c5aa81843a9b9a.json ./
RUN pip install -r requirements.txt
RUN pip freeze
CMD ["streamlit", "run", "tagging_app.py"]
EOF

set -eEx

docker build -t dataset-tagger .
