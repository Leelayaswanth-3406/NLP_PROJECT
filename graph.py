import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
import spacy
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
regex_tokenizer = RegexpTokenizer(r'\w+')
treebank_tokenizer = TreebankWordTokenizer()

def run_graph(text_list):

    nltk = tree = regex = spacy_c = bert = char = 0

    for text in text_list:
        nltk += len(word_tokenize(text))
        tree += len(treebank_tokenizer.tokenize(text))
        regex += len(regex_tokenizer.tokenize(text))
        spacy_c += len([t.text for t in nlp(text)])
        bert += len(bert_tokenizer.tokenize(text))
        char += len(list(text))

    names = ["NLTK", "Treebank", "Regex", "SpaCy", "BERT", "Char"]
    values = [nltk, tree, regex, spacy_c, bert, char]

    plt.bar(names, values)
    plt.title("Tokenizer Comparison")
    plt.xlabel("Tokenizer")
    plt.ylabel("Token Count")

    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center')

    plt.show()