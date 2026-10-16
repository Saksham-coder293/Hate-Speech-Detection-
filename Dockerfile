FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 NLTK_DATA=/usr/share/nltk_data

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data stopwords

COPY preprocess.py predict.py demo.py app.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
