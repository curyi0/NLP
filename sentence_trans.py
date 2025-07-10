import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
from sklearn.manifold import TSNE
if __name__== "__main__":
# 자동 임베딩
    model= SentenceTransformer('jhgan/ko-sroberta-multitask')

    sentences = [
        "오늘 날씨 정말 좋네요!",
        "이 영화는 너무 슬펐어요.",
        "맛있는 점심을 먹으러 갑시다.",
        "프로젝트 마감이 얼마 남지 않아 걱정입니다."
    ]
    raw_embedding= model.encode(sentences)
    # print(raw_embedding.size, sep="[]")

# 임베딩  표준화   평균(mean) =0,  표준편차=1 로 스케일링
    scaler=StandardScaler()
    std_embedding= scaler.fit_transform(raw_embedding)

    print(np.mean(raw_embedding) , '\n', np.std(raw_embedding))
    # print(np.mean(std_embedding), '')
    
    plt.show()