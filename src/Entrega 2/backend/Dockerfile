FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any needed
RUN apt-get update && apt-get install -y gcc libpq-dev

COPY requirements.txt .
COPY bd/requirements.txt bd/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r bd/requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
