import pandas as pd
import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from ollama import Client
import logging
import json
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
MODEL_NAME = "llava"
OLLAMA_URL = "http://localhost:11435"

# --- NLTK Setup ---
def ensure_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# --- Dataset Loader ---
def load_tickets():
    from data import load_dataset
    return load_dataset()

# --- Preprocessor ---
def build_preprocessor():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)

    def preprocess(text: str) -> str:
        text = text.lower().translate(translator)
        tokens = [w for w in text.split() if w and w not in stop_words]
        return ' '.join(lemmatizer.lemmatize(tok) for tok in tokens)

    return preprocess

# --- TF-IDF Index Builder ---
def build_chat_index(df: pd.DataFrame, preprocess_fn):
    required_fields = ['Product Purchased', 'Ticket Subject', 'Ticket Description']
    for field in required_fields:
        if field not in df.columns:
            df[field] = ''

    df['text'] = df['Product Purchased'].fillna('') + ' | ' + \
                 df['Ticket Subject'].fillna('') + ' | ' + \
                 df['Ticket Description'].fillna('')
    df['clean_text'] = df['text'].apply(preprocess_fn)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    return df, vectorizer, tfidf_matrix

# --- Ollama Client ---
def build_ollama_client(url: str) -> Client:
    return Client(host=url)

# --- Ollama Response Generator ---
def rewrite_with_ollama(ollama_client: Client, sentence: str, ticket_info: pd.Series, history: List[dict]) -> str:
    messages = [{'role': 'system', 'content': 'You are a helpful technical support assistant.'}]

    if ticket_info is not None:
        prompt_lines = [
            "Please draft a friendly, conversational support reply using the details below.",
            "Include subject, description, and any resolution only if the product name appears in the user query.",
            "",
            "Ticket Details:",
            f"- Product: {ticket_info.get('Product Purchased', 'N/A')}",
            f"- Subject: {ticket_info.get('Ticket Subject', 'N/A')}",
            f"- Description: {ticket_info.get('Ticket Description', 'N/A')}",
        ]

        prompt_lines += [
            "",
            "Response guidelines:",
            "- Begin with an apology for the inconvenience.",
            "- Offer to escalate to advanced support if needed.",
            "-Talk like u tell someone a problem if they are relax"
        ]

        messages.append({'role': 'system', 'content': "\n".join(prompt_lines)})

    if isinstance(history, list):
        for msg in history:
            if isinstance(msg, dict) and msg.get('role') in ['user', 'assistant'] and 'content' in msg:
                messages.append({'role': msg['role'], 'content': msg['content']})

    messages.append({'role': 'user', 'content': sentence})
    print(json.dumps(messages, indent=4))

    try:
        resp = ollama_client.chat(model=MODEL_NAME, messages=messages, stream=False)
        return resp.get('message', {}).get('content', "Unable to generate a detailed response at the moment.")
    except Exception as e:
        logging.error("Ollama Error: %s: %s", type(e).__name__, e)
        return "Unable to generate a detailed response at the moment. Please provide more information."

# --- Response Logic with History ---
def get_response_factory(df, vectorizer, tfidf_matrix, preprocess_fn, ollama_client, threshold: float = 0.2):
    def get_response(user_query, history):
        clean_q = preprocess_fn(user_query)
        sims = cosine_similarity(vectorizer.transform([clean_q]), tfidf_matrix).flatten()
        idx, score = sims.argmax(), sims.max()

        if score < threshold or len(history) > 0:
            reply = rewrite_with_ollama(ollama_client, user_query, None, history)
        else:
            record = df.iloc[idx]
            reply = rewrite_with_ollama(ollama_client, user_query, record, history)

        return [{"role": "assistant", "content": reply}]

    return get_response

# --- Main Gradio App ---
def main():
    ensure_nltk_data()
    df = load_tickets()

    drop_cols = [
        'Customer Name', 'Customer Email', 'Customer Age', 'Customer Gender',
        'Date of Purchase', 'Time to Resolution',
        'Customer Satisfaction Rating', 'First Response Time'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    examples = [
        "I have problem with dell xps",
        "Summarize this ticket in one sentence",
        "Is this a technical, billing, or other type of issue?",
        "Estimate how long this ticket might take to resolve"
    ]

    preprocess_fn = build_preprocessor()
    df, vectorizer, tfidf_matrix = build_chat_index(df, preprocess_fn)
    ollama_client = build_ollama_client(OLLAMA_URL)
    get_response = get_response_factory(df, vectorizer, tfidf_matrix, preprocess_fn, ollama_client)

    gr.ChatInterface(title='Customer Support ChatBot',
                     fn=get_response,
                     type="messages",
                     examples=examples
                     ).launch(share=False)

if __name__ == '__main__':
    main()
