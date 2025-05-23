import os
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from transformers import pipeline
import stanza
import requests

#initialize models and dataframe
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
stanza.download('en')
nlp = stanza.Pipeline('en')
data_columns = [
    "prompt", "prompt_timestamp", "word_count", "char_count", "prompt_emotion", "key_topics", 
    "deleted_words_percent", "added_words_num", "semantic_similarity", "change_summary"
]
data = pd.DataFrame(columns=data_columns)

def extract_keywords_stanza(text):#extracts noun phrases
    doc = nlp(text)
    keywords = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in {"NOUN", "PROPN"}:# geographic names?
                keywords.append(word.text)
    return list(set(keywords))

def calculate_semantic_similarity_stanza(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    #lemmatized token extraction from prompts (document)
    tokens1 = [word.lemma for sentence in doc1.sentences for word in sentence.words]
    tokens2 = [word.lemma for sentence in doc2.sentences for word in sentence.words]
    common_tokens = set(tokens1).intersection(tokens2)
    total_tokens = set(tokens1).union(tokens2)

    #Jaccard similarity
    return len(common_tokens) / len(total_tokens) if total_tokens else 0

def prompt_analysis(prompt):
    global data
    prompt_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    word_count = len(prompt.split())
    char_count = len(prompt)
    prompt_emotion = classifier(prompt)[0]['label']
    key_topics = extract_keywords_stanza(prompt)
    deleted_words_percent = 0
    added_words_num = 0
    semantic_similarity = 0
    change_summary = ""

    if data.empty:
        change_summary = "This is the first prompt. " + model.generate_content(
            ["Return a 30 word summary without commas of the Prompt. Prompt: '" + prompt + "'"],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        ).text.replace(",", " ")
    else:
        previous_prompt = data.iloc[-1]["prompt"]
        previous_prompt_words = set(previous_prompt.split())
        current_prompt_words = set(prompt.split())
        deleted_words = previous_prompt_words - current_prompt_words
        added_words = current_prompt_words - previous_prompt_words
        if previous_prompt_words:
            deleted_words_percent = round((len(deleted_words) / len(previous_prompt_words)) * 100, 2)
        added_words_num = len(added_words)
        semantic_similarity = calculate_semantic_similarity_stanza(previous_prompt, prompt)
        change_summary = model.generate_content(
            ["Return a roughly 30 word summary without commas of the change from previous prompt to current prompt. previous: '" + previous_prompt + "' current: '" + prompt + "'"],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        ).text.replace(",", " ")
#add results
    new_row = pd.DataFrame({
        "prompt": [prompt],
        "prompt_timestamp": [prompt_timestamp],
        "word_count": [word_count],
        "char_count": [char_count],
        "prompt_emotion": [prompt_emotion],
        "key_topics": [key_topics],
        "deleted_words_percent": [deleted_words_percent],
        "added_words_num": [added_words_num],
        "semantic_similarity": [semantic_similarity],
        "change_summary": [change_summary]
    })
    data = pd.concat([data, new_row], ignore_index=True)

def save_prompt_history():
    file_path = os.path.join("./", "prompt_history.csv")
    data.to_csv(file_path, index=False, encoding='utf-8')


'''
#emotion_conf_score = classifier(prompt)[0]['score']
#language = langdetect.detect(prompt)
#sentiment_polarity = TextBlob(prompt).sentiment.polarity
#polarity = "Positive" if sentiment_polarity > 0 else "Negative" if sentiment_polarity < 0 else "Neutral"
#sentiment_subjectivity = TextBlob(prompt).sentiment.subjectivity
#subjectivity = "Objective" if sentiment_subjectivity < 0.5 else "Subjective"
#num_complex_sentences = sum(1 for sent in nlp(prompt).sents if any(tok.dep_ in {"advcl", "ccomp", "xcomp", "relcl"} for tok in sent))
#grammar_mistakes = len(grammar.check(prompt))
#enhancing_request = any(any(keyword in key_topics for keyword in row_key.split(", "))for row_key in data["key_topics"].fillna("").tolist())
#new_request = not enhancing_request
'''