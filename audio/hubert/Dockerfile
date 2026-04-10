FROM pytorch/pytorch:2.3.1-cpu
WORKDIR /app

# Copy hubert service files
COPY . .

# Install Python dependencies if present
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Optional: set HF token for model downloads
ENV HF_TOKEN=hf_your_token_here

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
