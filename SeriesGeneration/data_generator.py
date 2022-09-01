# =============================================================================
#  libraries
# =============================================================================
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
logging.basicConfig(level=logging.INFO)
import word2vec as w2v
import pandas as pd

# =============================================================================
#  word2vec
# =============================================================================
def train_word2vec_model(input_file_path, output_file_path):
    sentences = LineSentence(input_file_path)
    model = Word2Vec(sentences=sentences, size=300, window=5, min_count=1, workers=2)
    model.wv.save_word2vec_format(output_file_path, binary=False)


def test_word2vec_model(input_file_path):
    wiki_model = KeyedVectors.load_word2vec_format(input_file_path)
    most_similar = wiki_model.most_similar(u'قاره')
    for words in most_similar:
        print(words[0])

# =============================================================================
#  format data
# =============================================================================
def format_data(input_file_path):
    result_file = open(CSV_FILE_NAME, 'w')
    with open(input_file_path, 'r') as file:
        # reading each line
        i = 1
        for line in file:
            # reading each word
            embedding_values = line.split()
            for j in range(1, len(embedding_values)):
                print(i, " -> ", embedding_values[j])
            i = i + 1
# =============================================================================
#  main
# =============================================================================
if __name__ == "__main__":
sample_persian_input_file = ""
embedding_output_file = ""
parsed_data_file = ""

w2v.train_word2vec_model(sample_persian_input_file, embedding_output_file)

cs.create_series(embedding_output_file,"",)