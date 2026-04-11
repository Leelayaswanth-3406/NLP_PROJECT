from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
import spacy
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
regex_tokenizer = RegexpTokenizer(r'\w+')
treebank_tokenizer = TreebankWordTokenizer()


def run_tokenization(text_list, choice):

    # Store all tokens
    all_tokens = {
        "NLTK": [],
        "Treebank": [],
        "Regex": [],
        "SpaCy": [],
        "BERT": [],
        "Character": []
    }

    print("\nProcessing Multiple Sentences...\n")

    for i, text in enumerate(text_list):

        print("\n" + "="*60)
        print(f"--- Sentence {i+1} ---\n{text}")

        # NLTK
        if choice == "1" or choice == "7":
            tokens = word_tokenize(text)
            all_tokens["NLTK"].extend(tokens)
            print("\nNLTK Tokens:\n", tokens)

        # Treebank
        if choice == "2" or choice == "7":
            tokens = treebank_tokenizer.tokenize(text)
            all_tokens["Treebank"].extend(tokens)
            print("\nTreebank Tokens:\n", tokens)

        # Regex
        if choice == "3" or choice == "7":
            tokens = regex_tokenizer.tokenize(text)
            all_tokens["Regex"].extend(tokens)
            print("\nRegex Tokens:\n", tokens)

        # SpaCy
        if choice == "4" or choice == "7":
            tokens = [t.text for t in nlp(text)]
            all_tokens["SpaCy"].extend(tokens)
            print("\nSpaCy Tokens:\n", tokens)

        # BERT
        if choice == "5" or choice == "7":
            tokens = bert_tokenizer.tokenize(text)
            all_tokens["BERT"].extend(tokens)
            print("\nBERT Tokens:\n", tokens)

        # Character
        if choice == "6" or choice == "7":
            tokens = list(text)
            all_tokens["Character"].extend(tokens)
            print("\nCharacter Tokens:\n", tokens)

    # ================== TOTAL COUNTS ==================

    print("\n==============================")
    print("TOTAL TOKEN COUNTS")
    print("==============================")

    for k, v in all_tokens.items():
        if (choice == "7") or (choice == str(list(all_tokens.keys()).index(k)+1)):
            print(f"{k:<10}: {len(v)}")

    # ================== COMPARISON TABLE ==================

    if choice == "7":

        print("\n==============================")
        print("TOKEN COMPARISON (FIRST 50)")
        print("==============================")

        limit = min(len(v) for v in all_tokens.values())
        limit = min(limit, 50)

        print(f"{'Index':<5} {'NLTK':<12} {'Treebank':<12} {'Regex':<12} {'SpaCy':<12} {'BERT':<12} {'Char':<8}")
        print("-" * 100)

        for i in range(limit):
            print(f"{i+1:<5} "
                  f"{all_tokens['NLTK'][i]:<12} "
                  f"{all_tokens['Treebank'][i]:<12} "
                  f"{all_tokens['Regex'][i]:<12} "
                  f"{all_tokens['SpaCy'][i]:<12} "
                  f"{all_tokens['BERT'][i]:<12} "
                  f"{all_tokens['Character'][i]:<8}")

        # ================== ANALYSIS ==================
        print("\nAnalysis:")
        print("NLTK → Basic tokenization")
        print("Treebank → Better punctuation handling")
        print("Regex → Removes punctuation")
        print("SpaCy → Linguistic rule-based")
        print("BERT → Subword tokenization")
        print("Character → Character-level analysis")