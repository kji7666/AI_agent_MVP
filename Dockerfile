# 使用 Python 3.11
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 先複製 requirements 並安裝 (利用 Docker Cache 加速)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼 (.env, json, py) 到容器內
COPY . .

# 預設執行指令
CMD ["python", "main.py"]