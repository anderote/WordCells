from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import openai
import threading
from threading import Thread
from queue import Queue


app = Flask(__name__)


# Add a function to load the API key with a timeout
def load_api_key_with_timeout(file_path, timeout=30):
    api_key = None

    def load_key():
        nonlocal api_key
        with open(file_path, 'r') as f:
            api_key = f.read().strip()

    load_key_thread = threading.Thread(target=load_key)
    load_key_thread.start()
    load_key_thread.join(timeout)

    if load_key_thread.is_alive():
        app.logger.error("Loading API key timed out.")
        load_key_thread.join()  # Ensure the thread is joined even after timeout
        return None

    return api_key

# Load the API key with a timeout
api_key_path = 'api_key.txt'
openai.api_key = load_api_key_with_timeout(api_key_path)

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]
    return words

def load_wordlists():
    wordlists = {"synonyms": {}, "antonyms": {}}
    for w_type in wordlists.keys():
        with open(f"wordlists/{w_type}.txt", "r") as file:
            for line in file:
                word, answers = line.strip().split("|")
                wordlists[w_type][word] = answers.split(",")
    return wordlists



glove_embeddings_path = 'glove.6B.50d.txt'
common_nouns_path = 'wordlists/nouns.txt'
common_nouns2_path = 'wordlists/nouns2.txt'
antonyms_path = 'wordlists/antonyms.txt'
synonyms_path = 'wordlists/synonyms.txt'

wordlists = load_wordlists()
common_nouns2 = load_words(common_nouns2_path)
common_nouns = load_words(common_nouns_path)
antonyms = load_words(antonyms_path)
synonyms = load_words(synonyms_path)
glove_model = load_glove_embeddings(glove_embeddings_path)




@app.route("/")
def index():
    return render_template("index.html")

# Add the following import
from itertools import combinations

@app.route("/generate_puzzle")
def generate_puzzle():
    difficulty = int(request.args.get('difficulty', 2))
    if difficulty < 2 or difficulty > 4:
        difficulty = 2

    # TODO: implement more types of puzzle formats, i.e. verb + nouns2, or noun + nouns2, synonyms, antonyms
    words = random.sample([word for word in common_nouns2 if word in glove_model], difficulty-1)
    primary_noun = random.sample(common_nouns, 1)[0]
    words.append(primary_noun)

    result_embedding = np.sum([glove_model[word] for word in words], axis=0)

    # Spawn a thread to find the correct answer
    correct_answer_queue = Queue()
    threading.Thread(target=find_correct_answer, args=(result_embedding, words, correct_answer_queue)).start()

    correct_answer = correct_answer_queue.get()

    return jsonify({"puzzle": " + ".join(words), "result_embedding": result_embedding.tolist(), "correct_answer": correct_answer})


# Define a function to find the correct answer
def find_correct_answer(result_embedding, puzzle_words, queue):
    max_similarity = -1
    correct_answer = None

    for word in common_nouns:
        if word in glove_model and word not in puzzle_words:
            similarity = cosine_similarity([glove_model[word]], [result_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                correct_answer = word

    queue.put(correct_answer)

@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity():
    user_words = request.form["user_words"]
    user_words = json.loads(user_words)  # Deserialize the JSON string
    result_embedding = np.array(request.form.getlist("result_embedding[]"), dtype=np.float32)

    puzzle_words = request.form["puzzle_words"].split(" + ")  # Get the puzzle words and split them

    user_embeddings = []
    matched_words = 0

    for user_word in user_words:
        if user_word in glove_model:
            if user_word not in puzzle_words:
                user_embeddings.append(glove_model[user_word])
            else:
                matched_words += 1
        else:
            return jsonify({"error": f"Invalid input: {user_word} is not in the vocabulary."})

    if matched_words == len(user_words):
        similarity = 0.0
    else:
        user_combined_embedding = np.sum(user_embeddings, axis=0)
        similarity = cosine_similarity([user_combined_embedding], [result_embedding])[0][0]
        similarity = float(np.round(similarity, 2))

    difficulty = len(puzzle_words) - 1
    if similarity > 0.5:
        score = difficulty
    elif similarity < -0.2:
        score = -difficulty / 2
    else:
        score = 0

    return jsonify({"similarity": round(similarity, 2), "score": score})



# Add a function to generate a poem using ChatGPT
def generate_poem(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


# Add a new route to generate a two-line poem when the puzzle generates
@app.route("/generate_poem_on_puzzle", methods=["POST"])
def generate_poem_on_puzzle():
    puzzle_words = request.form["puzzle_words"]
    prompt = f"Write me a two-line poem with rhyme scheme AB that includes the following words: {puzzle_words}"
    poem = 'poem goes here'
    #poem = generate_poem(prompt)
    return jsonify({"poem": poem})

# Add a new route to finish the poem when the user submits their words
@app.route("/generate_poem_on_submit", methods=["POST"])
def generate_poem_on_submit():
    user_words = request.form["user_words"]
    starting_poem = request.form["starting_poem"]
    prompt = f"Finish the following two-line poem with rhyme scheme AB by adding two more lines with rhyme scheme AB that include the following words: {user_words}\n\n{starting_poem}"
    finished_poem = ' made by gpt'
    # finished_poem = generate_poem(prompt)
    return jsonify({"poem": finished_poem})



if __name__ == "__main__":
    # exec(open("github_repo_summary.py").read())
    app.run(debug=True, port=5005)