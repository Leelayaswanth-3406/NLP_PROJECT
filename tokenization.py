from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def run_tokenization(text):

    nltk_tokens = word_tokenize(text)
    spacy_tokens = [t.text for t in nlp(text)]
    bert_tokens = bert_tokenizer.tokenize(text)

    print("\nSample Text:\n")
    print(text[:200])

    print("\nWord Tokenization (NLTK):\n")
    print(nltk_tokens)

    print("\nToken Counts:")
    print("NLTK :", len(nltk_tokens))
    print("SpaCy:", len(spacy_tokens))
    print("BERT :", len(bert_tokens))

    print("\nToken Comparison:\n")

    print(f"{'Index':<5} {'NLTK':<15} {'SpaCy':<15} {'BERT':<15}")
    print("-" * 55)

    for i in range(10):
        print(f"{i+1:<5} {nltk_tokens[i]:<15} {spacy_tokens[i]:<15} {bert_tokens[i]:<15}")

    print("\nDifference:")
    print("NLTK → Word-based")
    print("SpaCy → Linguistic")
    print("BERT → Subword (##)")

    print("\nAnalysis:")
print("BERT produces more tokens because it uses subword tokenization.")
print("NLTK and SpaCy use word-level tokenization.")