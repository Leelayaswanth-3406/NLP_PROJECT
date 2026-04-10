from conllu import parse
from tokenization import run_tokenization
from graph import run_graph
from dependency import run_dependency

# load dataset
with open("en_ewt-ud-train.conllu", "r", encoding="utf-8") as f:
    data = f.read()

sentences = parse(data)

# take few sentences
text_data = []
for sentence in sentences[:5]:
    words = [token["form"] for token in sentence]
    text_data.append(" ".join(words))

text = " ".join(text_data)

# menu
print("\n1. Tokenization")
print("2. Graph Comparison")
print("3. Dependency Parsing")

choice = input("\nEnter choice: ")

if choice == "1":
    run_tokenization(text)

elif choice == "2":
    run_graph(text)

elif choice == "3":
    run_dependency(text_data[0])