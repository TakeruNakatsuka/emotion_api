from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import librosa
import numpy as np
import shutil
import os
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Emotion API (Cloud Storage Mode) is running! 🚀"}

# ==========================================
# ☁️ Supabase（外部ストレージ）の設定
# ==========================================
# Renderの環境変数からキーを読み込む（見つからない場合は直書きのキーを使う）
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "ai-models"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = "sgd_model.pkl"
SCALER_PATH = "scaler.pkl"

# ==========================================
# 1. サーバー起動時にクラウドから最新の「脳みそ」と「定規」をダウンロード
# ==========================================
print("☁️ Supabaseから最新のAIモデルをダウンロードしています...")
try:
    # モデルのダウンロード
    with open(MODEL_PATH, "wb") as f:
        res = supabase.storage.from_(BUCKET_NAME).download(MODEL_PATH)
        f.write(res)
    # 定規のダウンロード
    with open(SCALER_PATH, "wb") as f:
        res = supabase.storage.from_(BUCKET_NAME).download(SCALER_PATH)
        f.write(res)
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("🎉 クラウドAIモデルの読み込みに完全成功しました！")
except Exception as e:
    print(f"⚠️ クラウドからのダウンロードに失敗しました: {e}")
    # フォールバック（ローカルにファイルがあればそれを使う）
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("💡 ローカルのAIモデルを読み込みました。")
    except:
        model = None
        scaler = None

# ==========================================
# 2. 感情判定エンドポイント（通常モード）
# ==========================================
@app.post("/analyze-emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    if model is None or scaler is None:
        return {"status": "error", "message": "AIモデルまたは定規が読み込まれていません。"}

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        y, sr = librosa.load(temp_file_path, sr=None, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        mfcc_scaled = scaler.transform([mfcc_mean])
        mfcc_scaled_32 = mfcc_scaled.astype(np.float32)

        prediction = model.predict(mfcc_scaled_32)
        result_emotion = prediction[0]

        return {
            "status": "success",
            "filename": file.filename,
            "emotion": result_emotion
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ==========================================
# 3. 追加学習 ＆ クラウドへ上書き保存エンドポイント
# ==========================================
@app.post("/feedback/")
async def save_feedback(
    file: UploadFile = File(...),
    correct_emotion: str = Form(...)
):
    if model is None or scaler is None:
        return {"status": "error", "message": "AIモデルがありません。"}

    temp_file_path = f"temp_feedback_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        y, sr = librosa.load(temp_file_path, sr=None, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        mfcc_scaled = scaler.transform([mfcc_mean])
        mfcc_scaled_32 = mfcc_scaled.astype(np.float32)

        # 今の脳みそを維持したまま「1件だけ」追加学習する！
        model.partial_fit(mfcc_scaled_32, [correct_emotion])

        # 1. まず Render のローカルサーバーに上書き保存
        joblib.dump(model, MODEL_PATH)

        # 2. ☁️ Supabase に賢くなった新しい脳みそをアップロード（上書き）する！
        supabase.storage.from_(BUCKET_NAME).upload(
            file=MODEL_PATH, 
            path=MODEL_PATH, 
            file_options={"upsert": "true"}
        )

        return {
            "status": "success",
            "message": f"学習完了！新しい脳みそをクラウド(Supabase)に上書き保存しました！🧠☁️✨"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)