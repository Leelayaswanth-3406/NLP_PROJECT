from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def run_tokenization(text_list, choice):

    all_nltk = []
    all_spacy = []
    all_bert = []

    print("\nProcessing Multiple Sentences...\n")

    for i, text in enumerate(text_list):
        print(f"\n--- Sentence {i+1} ---\n")
        print(text)

        if choice == "1" or choice == "4":
            nltk_tokens = word_tokenize(text)
            all_nltk.extend(nltk_tokens)
            print("\nNLTK Tokens:\n", nltk_tokens)

        if choice == "2" or choice == "4":
            spacy_tokens = [t.text for t in nlp(text)]
            all_spacy.extend(spacy_tokens)
            print("\nSpaCy Tokens:\n", spacy_tokens)

        if choice == "3" or choice == "4":
            bert_tokens = bert_tokenizer.tokenize(text)
            all_bert.extend(bert_tokens)
            print("\nBERT Tokens:\n", bert_tokens)

    # TOTAL COUNTS
    print("\n==============================")
    print("TOTAL TOKEN COUNTS")
    print("==============================")

    if choice == "1" or choice == "4":
        print("NLTK :", len(all_nltk))
    if choice == "2" or choice == "4":
        print("SpaCy:", len(all_spacy))
    if choice == "3" or choice == "4":
        print("BERT :", len(all_bert))

    # COMPARISON (FIRST 50)
    if choice == "4":
        print("\n==============================")
        print("TOKEN COMPARISON (FIRST 50)")
        print("==============================")

        limit = min(50, len(all_nltk), len(all_spacy), len(all_bert))

        print(f"{'Index':<5} {'NLTK':<15} {'SpaCy':<15} {'BERT':<15}")
        print("-" * 60)

        for i in range(limit):
            print(f"{i+1:<5} {all_nltk[i]:<15} {all_spacy[i]:<15} {all_bert[i]:<15}")

        print("\nAnalysis:")
        print("BERT produces more tokens due to subword tokenization.")