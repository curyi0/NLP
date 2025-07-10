##Fast API 생성하기   pip install fastapi "uvicorn[standard]"
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch, pickle, os
from fastapi.middleware.cors import CORSMiddleware

app= FastAPI() # 앱 생성
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000/"] 등 프론트 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용: GET, POST, OPTIONS 등
    allow_headers=["*"],  # 모든 헤더 허용
)
model_Path= 'venv/save_model'

class SentimentRequest(BaseModel):
    text: str

def make_classifier():
    try:  # GPU가 존재 유무에 따른 사용 결정
        device= 0 if torch.cuda.is_available() else -1
        sentiment_classifier=pipeline(
            'sentiment-analysis',
            model=model_Path,
            tokenizer=model_Path,
            device=device,  # CPU or GPU

        )
    except Exception as err:
        print(f"모델 로드 중 에러 발생 {err}")
        sentiment_classifier=None

    return sentiment_classifier

sentiment_classifier=None

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    sentiment_classifier= make_classifier()

    if sentiment_classifier is None:
        return {"error": " 서버 내부 오류;; 모델 로드 되지 않음"}
    try:
        result= sentiment_classifier( request.text)
        return result[0]
    except Exception as err:
        return {"error": f"예측 중 에러 발생: {err}"}

# get 방식 확인용
@app.get("/predict")
def read_get():

    return {"status":"sentiment analysis"} # 실행 구문
#  uvicorn FastAPI:app --reload   api 서버 실행
if __name__=="__main__":
    pass