from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2


app = Flask(__name__)
CORS(app)

conn_params = {
  "host": "localhost",
  "database": "mydb",
  "user": "oreo",
  "password": "1111"
}

def load_model(sentence):
    try:
        query = f'''SELECT document, similarity(lower('{sentence}'),
                       lower(document)) AS similarity_score
                       FROM cosine_similarity_table
                       ORDER BY similarity_score DESC
                       LIMIT 10;'''
        connection = psycopg2.connect(**conn_params)
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    except Exception as error:
        print(error)


@app.route("/predict", methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        text = data["text"]
        result = load_model(text)
        return jsonify({"label": result[0]})
    except Exception as error:
        return jsonify({"error": f"예측 중 에러 발생: {error}"})

@app.route("/predict", methods=["GET"])
def hello():
    return jsonify({"greet": "hello"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)