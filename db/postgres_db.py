from sqlalchemy import create_engine
import  warnings
warnings.warn("이건 무시됩니다.", DeprecationWarning)

# print("Hello!!") # recommand 파일에서 임포트후  실행하면 두 파일이 붙음,
#실행파일 => recommand , if __name__== "__main__":는 작성중인 곳 인식 "Hello!!"만 찍히고 아래실행x





def conn_postgres_db(df, id, pwd, db, table_name):
    # Corrected connection string
    url_conn = f"postgresql+psycopg2://{id}:{pwd}@localhost:5432/{db}"

    # Create the engine
    conn = create_engine(url_conn)

    # Write the DataFrame to PostgreSQL
    df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)


if __name__== "__main__":
    print("HI!")
    # conn_postgres_db(users_df, "oreo", 1111, "nmf_model")