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

def load_common_adjectives(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        adjectives = [line.strip() for line in f]
    return adjectives

def load_common_verbs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        verbs = [line.strip() for line in f]
    return verbs

glove_embeddings_path = 'glove.6B.50d.txt'
glove_model = load_glove_embeddings(glove_embeddings_path)
common_nouns_path = 'wordlists/nouns.txt'
common_nouns = load_common_nouns(common_nouns_path)
common_adjectives_path = 'wordlists/adjectives.txt'
common_adjectives = load_common_adjectives(common_adjectives_path)

@app.route("/")
def index():
    return render_template("index.html")

def generate_grammatical_puzzle():
    pattern = random.choice(["noun_adjective", "adjective_adjective", "noun_noun"])
    
    if pattern == "noun_adjective":
        noun = random.choice([word for word in common_nouns if word in glove_model])
        adjective = random.choice([word for word in common_adjectives if word in glove_model])
        result_embedding = glove_model[noun] + glove_model[adjective]
        puzzle = f"{noun} + {adjective}"
    if pattern == "noun_noun":
        noun_1 = random.choice([word for word in common_nouns if word in glove_model])
        noun_2 = random.choice([word for word in common_nouns if word in glove_model])
        result_embedding = glove_model[noun_1] + glove_model[noun_2]
        puzzle = f"{noun_1} + {noun_2}"
    else:  # pattern == "adjective_adjective"
        adjective_1 = random.choice([word for word in common_adjectives if word in glove_model])
        adjective_2 = random.choice([word for word in common_adjectives if word in glove_model])
        result_embedding = glove_model[adjective_1] + glove_model[adjective_2]
        puzzle = f"{adjective_1} + {adjective_2}"
        
    return puzzle, result_embedding

@app.route("/generate_puzzle")
def generate_puzzle():
    puzzle, result_embedding = generate_grammatical_puzzle()
    return jsonify({"puzzle": puzzle, "result_embedding": result_embedding.tolist()})

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
