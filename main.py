from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import shutil
import os
import joblib

app = FastAPI()

# サーバー起動時に、学習済みの本物AIモデルを読み込む
MODEL_PATH = "emotion_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("AIモデルの読み込みに成功しました！")
except Exception as e:
    print(f"エラー: {MODEL_PATH} が見つかりません。フォルダ内に配置しましたか？")
    model = None

@app.post("/analyze-emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    if model is None:
        return {"status": "error", "message": "AIモデルが読み込まれていません。"}

    # 送信された音声を一時ファイルとして保存
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. 音声ファイルを読み込む
        y, sr = librosa.load(temp_file_path, sr=None)
        
        # 2. 音声からAI用の特徴量（MFCC）を抽出する
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # 3. 本物のAIモデルを使って感情を推論する
        prediction = model.predict([mfcc_mean])
        result_emotion = prediction[0]

        # AIの確信度（スコア）を計算する
        probabilities = model.predict_proba([mfcc_mean])[0]
        confidence = round(float(max(probabilities)), 2)

        return {
            "status": "success",
            "filename": file.filename,
            "emotion": result_emotion,
            "confidence": confidence
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # 処理が終わったら一時ファイルを確実に削除する
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# クラウド環境やngrok環境でポート番号を自動割り当てするための設定
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)