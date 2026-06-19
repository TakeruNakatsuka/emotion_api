from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import librosa
import numpy as np
import shutil
import os
import joblib
import copy  # 🌟 初期AIを「コピー」して新しいユーザーに配るために追加

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
    return {"message": "AI Emotion API (Personalized Mode) is running! 🚀"}

# ==========================================
# ☁️ Supabase（外部ストレージ）の設定
# ==========================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "ai-models"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("\n🚨 [重大エラー] Renderの環境変数が設定されていません！")
    raise ValueError("Supabaseの環境変数が不足しています。")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 共通のファイル名
BASE_MODEL_PATH = "sgd_model.pkl"
SCALER_PATH = "scaler.pkl"

# ==========================================
# 1. サーバー起動時：共通の「初期脳みそ」と「定規」をロード
# ==========================================
try:
    with open(BASE_MODEL_PATH, "wb") as f:
        res = supabase.storage.from_(BUCKET_NAME).download(BASE_MODEL_PATH)
        f.write(res)
    with open(SCALER_PATH, "wb") as f:
        res = supabase.storage.from_(BUCKET_NAME).download(SCALER_PATH)
        f.write(res)
    
    base_model = joblib.load(BASE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("🎉 初期AIモデルと定規の準備が完了しました！")
except Exception as e:
    print(f"⚠️ 起動エラー（ローカルのファイルを使います）: {e}")
    base_model = joblib.load(BASE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# 🌟 ユーザー専用の脳みそをロードする便利関数
def get_user_model(user_id: str):
    user_model_path = f"sgd_model_{user_id}.pkl"
    
    # ① すでにRenderのローカルにダウンロード済みならそれを使う（爆速）
    if os.path.exists(user_model_path):
        return joblib.load(user_model_path), user_model_path
        
    # ② クラウド(Supabase)から探してダウンロードする
    try:
        res = supabase.storage.from_(BUCKET_NAME).download(user_model_path)
        with open(user_model_path, "wb") as f:
            f.write(res)
        return joblib.load(user_model_path), user_model_path
    except:
        # ③ 見つからない（新規ユーザー）場合は、初期AIをコピーしてプレゼントする！
        print(f"💡 新規ユーザー {user_id} です。初期モデルをコピーします。")
        return copy.deepcopy(base_model), user_model_path

# ==========================================
# 2. 感情判定エンドポイント（ユーザーID対応）
# ==========================================
@app.post("/analyze-emotion/")
async def analyze_emotion(
    file: UploadFile = File(...),
    user_id: str = Form(...)  # 🌟 名札を受け取る
):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        y, sr = librosa.load(temp_file_path, sr=None, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        mfcc_scaled = scaler.transform([mfcc_mean])
        mfcc_scaled_32 = mfcc_scaled.astype(np.float32)

        # 🌟 その人専用の脳みそをロードして判定！
        user_model, _ = get_user_model(user_id)
        prediction = user_model.predict(mfcc_scaled_32)

        return {"status": "success", "emotion": prediction[0], "user_id": user_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ==========================================
# 3. 追加学習 ＆ 上書き保存（ユーザーID対応）
# ==========================================
@app.post("/feedback/")
async def save_feedback(
    file: UploadFile = File(...),
    correct_emotion: str = Form(...),
    user_id: str = Form(...)  # 🌟 名札を受け取る
):
    temp_file_path = f"temp_feedback_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        y, sr = librosa.load(temp_file_path, sr=None, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        mfcc_scaled = scaler.transform([mfcc_mean])
        mfcc_scaled_32 = mfcc_scaled.astype(np.float32)

        # 🌟 その人専用の脳みそをロードして追加学習！
        user_model, user_model_path = get_user_model(user_id)
        user_model.partial_fit(mfcc_scaled_32, [correct_emotion])

        # Renderに保存
        joblib.dump(user_model, user_model_path)

        # Supabaseにその人の名前でアップロード！
        supabase.storage.from_(BUCKET_NAME).upload(
            file=user_model_path, 
            path=user_model_path, 
            file_options={"upsert": "true"}
        )

        return {
            "status": "success",
            "message": f"{user_id} さん専用のAIとして学習・保存しました！🧠✨"
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