from tokenization import run_tokenization
from graph import run_graph
from dependency import run_dependency
from conllu import parse

# Load dataset
with open("en_ewt-ud-train.conllu", "r", encoding="utf-8") as f:
    data = f.read()

sentences = parse(data)

# Extract 8 sentences
text_data = []
for sentence in sentences[:8]:
    words = [token["form"] for token in sentence]
    text_data.append(" ".join(words))

# MAIN MENU
print("\n1. Tokenization")
print("2. Graph Comparison")
print("3. Dependency Parsing")

main_choice = input("Enter choice: ")

# TOKENIZATION
if main_choice == "1":

    print("\nChoose Tokenizer:")
    print("1. NLTK")
    print("2. SpaCy")
    print("3. BERT")
    print("4. All (Comparison)")

    choice = input("Enter choice: ")

    run_tokenization(text_data, choice)

# GRAPH
elif main_choice == "2":
    run_graph(text_data)

# DEPENDENCY
elif main_choice == "3":
    run_dependency(text_data)

else:
    print("Invalid choice")