from fastapi import FastAPI, UploadFile, File, Form  # 🌟 Form を追加
import librosa
import numpy as np
import shutil
import os
import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "AI Emotion API (Online Learning Mode) is running! 🚀"}

# ==========================================
# 1. サーバー起動時に「脳みそ」と「定規」を読み込む
# ==========================================
MODEL_PATH = "sgd_model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("🎉 AIモデルと定規の読み込みに完全成功しました！")
except Exception as e:
    print(f"⚠️ エラー: {e}")
    print("フォルダ内に sgd_model.pkl と scaler.pkl は配置されていますか？")
    model = None
    scaler = None


# ==========================================
# 2. いつもの感情判定エンドポイント
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

        # 🌟 ここが超重要！ AIに渡す前に「定規」で数値を綺麗に整える
        mfcc_scaled = scaler.transform([mfcc_mean])

        # 判定
        prediction = model.predict(mfcc_scaled)
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
# 3. 🌟 新規追加！ ユーザー専用にチューニングするエンドポイント
# ==========================================
@app.post("/feedback/")
async def save_feedback(
    file: UploadFile = File(...),
    correct_emotion: str = Form(...)  # ユーザーからの「本当は〇〇だよ」という正解を受け取る
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

        # 🌟 ここでも定規で整える
        mfcc_scaled = scaler.transform([mfcc_mean])

        # 🧠 魔法の関数！今の脳みそを維持したまま「1件だけ」追加学習する！
        model.partial_fit(mfcc_scaled, [correct_emotion])

        # 💾 賢くなった脳みそを上書き保存する（次に聞かれた時は賢くなっている）
        joblib.dump(model, MODEL_PATH)

        return {
            "status": "success",
            "message": f"学習完了！AIの脳みそを「{correct_emotion}」に合わせてチューニングしました！🧠✨"
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