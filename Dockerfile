FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Chroma's local embedding model into the image so start-up needs no download.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"

COPY . .

ENV PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
