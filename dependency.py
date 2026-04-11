import spacy

nlp = spacy.load("en_core_web_sm")

def run_dependency(text_list):

    print("\nDependency Parsing:\n")

    text = text_list[0]

    doc = nlp(text)

    for token in doc:
        print(f"{token.text} ---> {token.dep_} ---> {token.head.text}")