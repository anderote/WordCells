from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_common_nouns(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nouns = [line.strip() for line in f]
    return nouns

glove_embeddings_path = 'glove.6B.50d.txt'  # Update this path to match the location of the downloaded file
glove_model = load_glove_embeddings(glove_embeddings_path)
common_nouns_path = 'common_nouns.txt'  # Update this path to match the location of the common_nouns.txt file
common_nouns = load_common_nouns(common_nouns_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_puzzle")
def generate_puzzle():
    words = random.sample([word for word in common_nouns if word in glove_model], 2)
    operator = random.choice(["+", "-"])

    if operator == "+":
        result_embedding = glove_model[words[0]] + glove_model[words[1]]
    else:
        result_embedding = glove_model[words[0]] - glove_model[words[1]]

    return jsonify({"puzzle": f"{words[0]} {operator} {words[1]}", "result_embedding": result_embedding.tolist()})

@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity():
    user_word = request.form["user_word"]
    result_embedding = np.array(request.form.getlist("result_embedding[]"), dtype=np.float32)

    if user_word in glove_model:
        user_embedding = glove_model[user_word]
        similarity = cosine_similarity([user_embedding], [result_embedding])[0][0]
        similarity = float(np.round(similarity, 2))
        return jsonify({"similarity": round(similarity, 2)})
    else:
        return jsonify({"error": "Invalid input. Please enter a word from the vocabulary."})

if __name__ == "__main__":
    app.run(debug=True)
