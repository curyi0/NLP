import pandas as pd
import pickle

from db.postgres_db import conn_postgres_db
##  코사인 유사도  cf) chat bot

if __name__=="__main__":
    with open("data/NLP_set.pkl", "rb") as f:
        data=pickle.load(f)

    df= pd.DataFrame(data['train'])
    print(df[ ["document", "label"]])  #  머신 돌리기 위해 필요한 정보들만
    conn_postgres_db(df, "oreo", '1111', 'mydb',"cosine_similarity_table")

    '''  유사도 찾기 쿼리문
    SELECT document, similarity(lower('{sentence}'),
               lower(document)) AS similarity_score
               FROM cosine_similarity_table
               ORDER BY similarity_score DESC
               LIMIT 10
    '''