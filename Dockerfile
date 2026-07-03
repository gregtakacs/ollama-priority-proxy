FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg2 lsb-release ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY OllamaModelProxy.py ./

EXPOSE 8080
CMD ["python3", "OllamaModelProxy.py"]
