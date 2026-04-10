import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def run_graph(text_list):

    nltk_count = 0
    spacy_count = 0
    bert_count = 0

    for text in text_list:
        nltk_count += len(word_tokenize(text))
        spacy_count += len([t.text for t in nlp(text)])
        bert_count += len(bert_tokenizer.tokenize(text))

    names = ["NLTK", "SpaCy", "BERT"]
    values = [nltk_count, spacy_count, bert_count]

    plt.bar(names, values)
    plt.title("Tokenizer Comparison")
    plt.xlabel("Tokenizer")
    plt.ylabel("Number of Tokens")

    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center')

    plt.show()