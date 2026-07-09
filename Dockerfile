# syntax=docker/dockerfile:1.6
FROM python:3.12-slim as builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.12-slim as runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY main.py .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "main.py"]
