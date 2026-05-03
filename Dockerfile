FROM python:3.10-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl libcrypt1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["/bin/sh", "-lc", "/app/.venv/bin/python api.py --host 0.0.0.0 --port 8000 --output-dir /data/output"]
