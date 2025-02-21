from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def analyze_sentiment(headline):
    # Tokenize the input headline
    inputs = tokenizer(headline, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Perform sentiment analysis using FinBERT
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get the predicted sentiment class (0=negative, 1=neutral, 2=positive)
    sentiment_class = torch.argmax(logits, dim=-1).item()
    
    sentiment = {0: "1", 1: "0", 2: "-1"}
    
    return sentiment[sentiment_class]



from pygooglenews import GoogleNews
import json
import time
import csv

year = "2024"


gn = GoogleNews()
top = gn.search('oil OR crude oil OR futures OR energy OR energy futures OR volitility OR opec OR supply chain -olive',
                 from_=f"{year}-1-1", to_=f"{year}-12-30")



with open(f'!{year}.csv', 'w', newline='', encoding="utf-8") as file:

  entries = top["entries"]
  count = 0
  writer = csv.writer(file)

  writer.writerow(["count", "sentiment", "date", "title"])
  

  for entry in entries:
    count = count + 1

    sentiment = analyze_sentiment(entry["title"])

    print(count,entry["published"], entry["title"])
    writer.writerow([count, sentiment, entry["published"], entry["title"]])