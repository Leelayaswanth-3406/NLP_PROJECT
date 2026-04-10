import spacy

nlp = spacy.load("en_core_web_sm")

def run_dependency(sentence):

    print("\nSentence:\n", sentence)

    doc = nlp(sentence)

    print("\nDependency Parsing:\n")

    for token in doc:
        print(f"{token.text} → {token.dep_} → {token.head.text}")