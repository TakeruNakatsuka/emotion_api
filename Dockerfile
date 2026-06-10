# 1. ベースとなるOS（Pythonが入った軽量なLinux）を用意
FROM python:3.9-slim

# 2. 🌟ここで「librosaの壁」を突破！音声処理用のシステム部品をインストール
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# 3. 作業用フォルダを準備
WORKDIR /app

# 4. pythonのライブラリ（requirements.txt）をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. あなたのコードとAIモデル（main.py や .pkl）をすべてコピー
COPY . .

# 6. FastAPIサーバーを起動（ポート番号はRenderが自動で割り当てるものを利用）
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
