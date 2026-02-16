# Version: 0.9.5
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT
FROM nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04

WORKDIR /app
COPY requirements/base.txt /app/requirements/base.txt
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -r /app/requirements/base.txt
COPY src /app/src
COPY traffic_ai_assist.py /app/traffic_ai_assist.py
CMD ["python3", "traffic_ai_assist.py", "--serve", "--host", "0.0.0.0", "--port", "8080"]
