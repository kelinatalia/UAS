FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app_streamlit_cloud.py", "--server.port=8501", "--server.address=0.0.0.0"]