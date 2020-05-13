import numpy as np
import pickle
from config import *


# function loads word vectors and stores them one into big matrix
def load_w2v_en_to_ge_embeddings(sorted_vocabulary, we_filepath, we_dimension):
    we_matrix = np.zeros(shape=(len(sorted_vocabulary), we_dimension))

    # create we_matrix
    with open(we_filepath, 'r', encoding="utf-8") as vector_file:
        print("  opening vector file...")
        for line in vector_file:
            k = 0
            number_of_words = we_word_count
            dimension = we_dimension
            data = vector_file.readlines()
            for line in data:
                dictionary_index = -1
                # prvni radku preskocit
                if k == 0:
                    k += 1
                    continue
                values = line.split(" ")
                word = values[0]
                # najdeme index slova ve slovniku
                word_index = -1
                if word in sorted_vocabulary:
                    word_index = sorted_vocabulary[word]

                # dimension = int(dimension)
                vector = np.array(values[1:dimension + 1], dtype="float")
                if word_index == -1:
                    continue
                if word_index < len(sorted_vocabulary):
                    we_matrix[word_index] = vector
                k += 1
                if k % 10000 == 0:
                    print("Zpracovano", k, "/", number_of_words)

    # musime jeste pridat do matice embeddingy padding a oov
    padding_dict_index = sorted_vocabulary[PADDING_SYMBOL]
    out_of_vocabulary_dict_index = sorted_vocabulary[OUT_OF_VOCABULARY_SYMBOL]
    we_matrix[padding_dict_index] = np.zeros(we_dimension)
    we_matrix[out_of_vocabulary_dict_index] = np.zeros(we_dimension)

    with open("w2v_we_en_to_ge_matrix.bin", "wb") as f:
        pickle.dump(we_matrix, f)

    print(we_matrix.shape, we_matrix)
    return we_matrix




# function loads word vectors and stores them one into big matrix
def load_w2v_ge_to_en_embeddings(sorted_vocabulary, we_filepath, we_dimension):
    we_matrix = np.zeros(shape=(len(sorted_vocabulary), we_dimension))

    # create we_matrix
    with open(we_filepath, 'r', encoding="utf-8") as vector_file:
        print("  opening vector file...")
        for line in vector_file:
            k = 0
            number_of_words = we_word_count
            dimension = we_dimension
            data = vector_file.readlines()
            for line in data:
                dictionary_index = -1
                # prvni radku preskocit
                if k == 0:
                    k += 1
                    continue
                values = line.split(" ")
                word = values[0]
                # najdeme index slova ve slovniku
                word_index = -1
                if word in sorted_vocabulary:
                    word_index = sorted_vocabulary[word]

                # dimension = int(dimension)
                vector = np.array(values[1:dimension + 1], dtype="float")
                if word_index == -1:
                    continue
                if word_index < len(sorted_vocabulary):
                    we_matrix[word_index] = vector
                k += 1
                if k % 10000 == 0:
                    print("Zpracovano", k, "/", number_of_words)

    # musime jeste pridat do matice embeddingy padding a oov
    padding_dict_index = sorted_vocabulary[PADDING_SYMBOL]
    out_of_vocabulary_dict_index = sorted_vocabulary[OUT_OF_VOCABULARY_SYMBOL]
    we_matrix[padding_dict_index] = np.zeros(we_dimension)
    we_matrix[out_of_vocabulary_dict_index] = np.zeros(we_dimension)

    with open("w2v_we_ge_matrix.bin", "wb") as f:
        pickle.dump(we_matrix, f)

    print(we_matrix.shape, we_matrix)
    return we_matrix

# function loads word vectors and stores them one into big matrix
def load_w2v_embeddings(sorted_vocabulary, we_filepath, we_dimension, output_path):
    we_matrix = np.zeros(shape=(len(sorted_vocabulary), we_dimension))

    # create we_matrix
    with open(we_filepath, 'r', encoding="utf-8") as vector_file:
        print("  opening vector file...")
        for line in vector_file:
            k = 0
            number_of_words = we_word_count
            dimension = we_dimension
            data = vector_file.readlines()
            for line in data:
                dictionary_index = -1
                # prvni radku preskocit
                if k == 0:
                    k += 1
                    continue
                values = line.split(" ")
                word = values[0]
                # najdeme index slova ve slovniku
                word_index = -1
                if word in sorted_vocabulary:
                    word_index = sorted_vocabulary[word]

                # dimension = int(dimension)
                vector = np.array(values[1:dimension + 1], dtype="float")
                if word_index == -1:
                    continue
                if word_index < len(sorted_vocabulary):
                    we_matrix[word_index] = vector
                k += 1
                if k % 10000 == 0:
                    print("Zpracovano", k, "/", number_of_words)

    # musime jeste pridat do matice embeddingy padding a oov
    padding_dict_index = sorted_vocabulary[PADDING_SYMBOL]
    out_of_vocabulary_dict_index = sorted_vocabulary[OUT_OF_VOCABULARY_SYMBOL]
    we_matrix[padding_dict_index] = np.zeros(we_dimension)
    we_matrix[out_of_vocabulary_dict_index] = np.zeros(we_dimension)

    with open(output_path, "wb") as f:
        pickle.dump(we_matrix, f)

    print(we_matrix.shape, we_matrix)
    return we_matrix



def save_bin_data(data, path):
    # save data to a file
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)
    print("Data successfully saved")


def load_bin_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data