import re
import os
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from tqdm.notebook import tqdm
tqdm.pandas()

#Load the pre-trained model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


word_lst = ['parking', 'Parking']

def split_comment(text):
    """Split the comment based on sentence stop sign"""
    text = text.strip()
    text = text.replace("\n", ". ")
    text = text.replace("\r", ". ")
    text = text.replace("\r\n", ". ")
    text = text.replace("----", ".")
    text = text.replace("...", ".")
    text_list = re.split(r'(?<=[.!?])\s+', text)
    text_list = [text for text in text_list if text.strip()]
    text_list = [text.strip() for text in text_list]
    return text_list

def process_comment(text):
    """Extract the sentence that meanions parking"""
    text_list = split_comment(text)
    text_list = [text for text in text_list if any(word in text for word in word_lst)]
    filtered_text = ' '.join(map(str, text_list))
    filtered_text = filtered_text.strip()
    return filtered_text

def roberta_polarity_scores(text):
    """Apply the roberta sentiment model"""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }
    return scores_dict

def roberta_sentiment(text):
    """Categorize the sentiment based on roberta sentiment polarity"""
    try:
        scores_dict = roberta_polarity_scores(text)
        sentiment = max(scores_dict, key=scores_dict.get)
        return sentiment
    except:
        return ""


if __name__ == "__main__":
    read_dir = "parking-review"
    save_dir = "parking-review-sentiment"
    os.makedirs(save_dir, exist_ok=True)
    files = [file for file in os.listdir(read_dir) if file.endswith('.csv')]
    files = sorted(files)
    for file in files:
        read_filepath = f"{read_dir}/{file}"
        save_filepath = f"{save_dir}/{file}"
        print(f"---- Start processing: {file}")
        reader = pd.read_csv(read_filepath, chunksize=10000)
        chunks = []
        for chunk in reader:
            chunk['parking_text'] = chunk.apply(lambda x: process_comment(str(x['text'])), axis=1)
            chunk['sentiment'] = chunk.progress_apply(lambda x: roberta_sentiment(str(x['parking_text'])), axis=1)
            chunks.append(chunk)
        df = pd.concat(chunks)
        df.to_csv(save_filepath, index=False)
