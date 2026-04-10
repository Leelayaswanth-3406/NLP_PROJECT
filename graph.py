from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def run_graph(text):

    nltk_tokens = word_tokenize(text)
    spacy_tokens = [t.text for t in nlp(text)]
    bert_tokens = bert_tokenizer.tokenize(text)

    names = ['NLTK', 'SpaCy', 'BERT']
    counts = [len(nltk_tokens), len(spacy_tokens), len(bert_tokens)]

    plt.bar(names, counts)
    plt.title("Tokenizer Comparison")
    plt.xlabel("Tokenizer")
    plt.ylabel("Number of Tokens")

    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha='center')

    plt.show()

    plt.savefig("tokenizer_comparison.png")

    plt.title("Tokenizer Comparison (NLTK vs SpaCy vs BERT)")
    plt.grid(axis='y')