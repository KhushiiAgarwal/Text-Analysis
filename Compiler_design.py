# !pip install language_tool_python
# !pip install spacy
# !pip install pydot
# !pip install torch
import matplotlib.pyplot as plt
import spacy
import language_tool_python
import pydot
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the spaCy English model
# !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Initialize LanguageTool
grammar_tool = language_tool_python.LanguageTool('en-US')
pretrained_model_name = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)

sentence = input("Enter a sentence: ")

txt = nlp(sentence)

graph = pydot.Dot(graph_type='digraph', bgcolor='white')

for token in txt:
    fillcolor = 'yellow' if token.head == token else 'lightblue'
    node = pydot.Node(token.text, style='filled', fillcolor=fillcolor)
    graph.add_node(node)

for token in txt:
    if token.dep_ != 'punct':  # Exclude punctuation
        head_token = token.head
        edge = pydot.Edge(head_token.text, token.text, label=token.dep_, color='red')
        graph.add_edge(edge)

image_bytes = graph.create_png(prog='dot')
#Display image
with BytesIO(image_bytes) as img_buffer:
    img = Image.open(img_buffer)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()

# Check for grammatical errors
errors = grammar_tool.check(sentence)
if errors:
    print("\nGrammatical Errors:")
    for error in errors:
        print(f"Error at position {error.offset}-{error.offset + error.errorLength}: {error.message}")
        print(f"Suggestions: {error.replacements}")
else:
    print("\nNo grammatical errors found.")

print("\nSemantic Analysis:")
for token in txt:
    print(f"Token: {token.text}    ||     Part of Speech: {token.pos_}")

# Perform Named Entity Recognition (NER)
print("\nNamed Entity Recognition:")
for ent in txt.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Load the pre-trained RoBERTa model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment(sentence):
    encoded_sentence = tokenizer(sentence, padding=True, return_tensors="pt")
    output = model(**encoded_sentence)
    scores = output.logits.detach().numpy()
    probabilities = torch.softmax(torch.tensor(scores), dim=-1).numpy()[0]
    sentiment_id = np.argmax(probabilities)
    sentiment_label = 'positive' if sentiment_id == 2 else 'negative' if sentiment_id == 0 else 'neutral'
    return sentiment_label

sentiment = analyze_sentiment(sentence)
print("Sentiment:", sentiment)

# # Sentiment analysis
# from textblob import TextBlob

# text_blob = TextBlob(sentence)
# sentiment = text_blob.sentiment

# print("\nSentiment Analysis:")
# if sentiment.polarity > 0 :
#     polarity="Positive"
# elif sentiment.polarity == 0:
#     polarity="Neutral"
# else:
#     polarity="negative"
# print(f"Polarity: {polarity}")
