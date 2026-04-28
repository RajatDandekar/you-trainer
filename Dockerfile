FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN pip install --no-cache-dir \
      "transformers>=4.45,<5" \
      "peft>=0.13" \
      "accelerate>=0.34" \
      "datasets>=2.20" \
      "huggingface_hub>=0.25" \
      "runpod>=1.7" \
      "bitsandbytes>=0.43" \
      "safetensors>=0.4"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
