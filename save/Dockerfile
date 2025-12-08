# 使用輕量級 Python 3.11
FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴 (ChromaDB 有時需要 build tools)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 設定 Python 路徑，確保可以 import src
ENV PYTHONPATH=/app

# 預設指令 (保持容器開啟)
CMD ["tail", "-f", "/dev/null"]