FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
      transformers>=4.45 \
      peft>=0.13 \
      accelerate>=0.34 \
      datasets>=2.20 \
      huggingface_hub>=0.25 \
      runpod>=1.7 \
      bitsandbytes>=0.43

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
