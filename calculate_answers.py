import numpy as np
from annoy import AnnoyIndex
import os

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def build_annoy_index(embeddings, dimensions):
    index = AnnoyIndex(dimensions, 'angular')
    word_to_index = {}
    index_to_word = {}
    
    for i, (word, vector) in enumerate(embeddings.items()):
        index.add_item(i, vector)
        word_to_index[word] = i
        index_to_word[i] = word
        
    index.build(50)
    return index, word_to_index, index_to_word

def solve_puzzles(input_file, output_file, embeddings, annoy_index, word_to_index, index_to_word):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            words = line.strip().split('+')
            vector_sum = np.zeros(50, dtype='float32')
            
            for word in words:
                if word in embeddings:
                    vector_sum += embeddings[word]

            # Find the 10 closest words
            closest_indices = annoy_index.get_nns_by_vector(vector_sum, 10)

            for idx in closest_indices:
                closest_word = index_to_word[idx]
                if closest_word not in words:
                    f_out.write(closest_word + '\n')
                    break




glove_model = load_glove_embeddings('glove.6B.50d.txt')
annoy_index, word_to_index, index_to_word = build_annoy_index(glove_model, 50)

solve_puzzles('wordlists/puzzles.txt', 'wordlists/answers.txt', glove_model, annoy_index, word_to_index, index_to_word)
