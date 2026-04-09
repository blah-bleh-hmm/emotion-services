FROM pytorch/pytorch:2.0-cuda11.8-runtime-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY utils.py .

# Download model from HF at build time (or runtime, your choice)
ENV HF_TOKEN=hf_your_token_here

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
