FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy wav2vec service files
COPY . .

# Install Python dependencies if present
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Optional: set HF token for model downloads
ENV HF_TOKEN=hf_your_token_here

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
