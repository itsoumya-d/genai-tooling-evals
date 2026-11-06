FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e . --no-cache-dir
EXPOSE 8000
CMD ["uvicorn", "genai_evals.app:app", "--host", "0.0.0.0", "--port", "8000"]
