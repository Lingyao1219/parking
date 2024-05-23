import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

filters = ['parking']
filters_pos = ['park', 'parked']


def check_filters(text):
    """Check if the text contains any of the parking filters."""
    txt = text.lower().strip()
    if any(f in txt for f in filters):
        return text
    elif any(f in txt.split() for f in filters_pos):
        tokens = word_tokenize(txt)
        pos_tags = pos_tag(tokens)
        for word, pos in pos_tags:
            if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ') and (word in filters_pos):
                return text
        return ""
    else:
        return ""


def split_comment(text):
    """Split the text into sentences."""
    text = text.strip()
    text = text.replace("\n", ". ")
    text = text.replace("\r", ". ")
    text = text.replace("\r\n", ". ")
    text = text.replace("----", ".")
    text = text.replace("...", ".")
    text = text.replace("(Translated by Google)", "")
    text_list = re.split(r'(?<=[.!?])\s+', text)
    text_list = [text for text in text_list if text.strip()]
    text_list = [text.strip() for text in text_list]
    return text_list


def process_comment(text):
    """Process the comment text and ensure that each is related to parking."""
    text_list = split_comment(text)
    word_lst = filters + filters_pos
    text_list = [text for text in text_list if any(word in text.lower() for word in word_lst)]
    text_list = [check_filters(text) for text in text_list]
    filtered_text = ' '.join(map(str, text_list))
    filtered_text = filtered_text.strip()
    return filtered_text   
