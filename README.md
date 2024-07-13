# Hybrid PDF Search System

This repository contains three Python scripts for performing text search in a PDF file using different methods: semantic search, TF-IDF-based textual search, and a hybrid approach combining both methods. Each script extracts text from all pages of a PDF file and searches for the most relevant page based on a given query.

## Table of Contents
- [Dependencies](#dependencies)
- [Semantic Search](#semantic-search)
- [Textual Search with TF-IDF](#textual-search-with-tfidf)
- [Hybrid Search](#hybrid-search)

## Semantic Search
This script uses a pre-trained transformer model to embed the text and perform a semantic search based on cosine similarity.

Import necessary libraries:

import fitz  # PyMuPDF for extracting text from PDFs
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

Define the function to extract text from the PDF:

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        all_text.append(text)
    return all_text
Define the function to embed text:

python
Copy code
def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings
Define the search function:

python
Copy code
def search(query, embeddings, documents, model, tokenizer, top_k=1):
    query_embedding = embed_text(query, model, tokenizer)
    similarities = [torch.cosine_similarity(query_embedding, doc_emb, dim=1).item() for doc_emb in embeddings]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    most_relevant_index = sorted_indices[0]
    most_relevant_page = most_relevant_index + 1  # Adjust for 1-based index
    answer_text = documents[most_relevant_index]
    return most_relevant_page, answer_text
Load the pre-trained model and tokenizer:

python
Copy code
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
Define the path to your PDF and extract text:

python
Copy code
pdf_path = '/content/DL BOOK .pdf'  # Replace with your actual PDF path
pdf_texts = extract_text_from_pdf(pdf_path)
Generate embeddings for each page's text:

python
Copy code
embeddings = [embed_text(text, model, tokenizer) for text in pdf_texts]
Perform a search with a query:

python
Copy code
query = "Single Computational Layer: The Perceptron"  # Replace with your desired query
most_relevant_page, answer_text = search(query, embeddings, pdf_texts, model, tokenizer)
print(f"Most Relevant Page: {most_relevant_page}")
print(f"Answer Text: {answer_text}")
Textual Search with TF-IDF
This script uses TF-IDF vectorization and cosine similarity to perform a textual search in a PDF.

Usage
Import necessary libraries:

python
Copy code
import fitz  # PyMuPDF for extracting text from PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Define the function to extract text from the PDF:

python
Copy code
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        all_text.append(text)
    return all_text
Define the search function using TF-IDF:

python
Copy code
def search_in_pdf(pdf_path, query):
    pdf_texts = extract_text_from_pdf(pdf_path)
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(pdf_texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_vectors).flatten()
    most_similar_index = similarities.argmax()
    most_relevant_page = most_similar_index + 1  # Adjust for 1-based index
    answer_text = pdf_texts[most_similar_index]
    return most_relevant_page, answer_text
Define the path to your PDF and perform a search with a query:

python
Copy code
pdf_path = '/content/DL BOOK .pdf'  # Replace with your actual PDF path
query = "Single Computational Layer: The Perceptron"
most_relevant_page, answer_text = search_in_pdf(pdf_path, query)
print(f"Most Relevant Page: {most_relevant_page}")
print(f"Answer Text: {answer_text}")
Hybrid Search
This script combines semantic search using a pre-trained transformer model and textual search using TF-IDF to provide a more robust search capability.

Usage
Import necessary libraries:

python
Copy code
import fitz  # PyMuPDF for extracting text from PDFs
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
Define the function to extract text from the PDF:

python
Copy code
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        all_text.append(text)
    return all_text
Define the function to embed text:

python
Copy code
def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings
Define the function to create TF-IDF vectors:

python
Copy code
def create_tfidf_vectors(texts):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_vectors
Define the hybrid search function:

python
Copy code
def hybrid_search_in_pdf(pdf_path, query, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    pdf_texts = extract_text_from_pdf(pdf_path)
    page_embeddings = [embed_text(text, model, tokenizer) for text in pdf_texts]
    tfidf_vectorizer, tfidf_vectors = create_tfidf_vectors(pdf_texts)
    query_embedding = embed_text(query, model, tokenizer)
    similarities = [torch.cosine_similarity(query_embedding, emb, dim=1).item() for emb in page_embeddings]
    semantic_scores = np.array(similarities)
    query_vector = tfidf_vectorizer.transform([query])
    tfidf_similarities = cosine_similarity(query_vector, tfidf_vectors).flatten()
    combined_scores = semantic_scores + tfidf_similarities
    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
    ranked_pages = [(idx + 1, pdf_texts[idx]) for idx in ranked_indices]  # Adjust for 1-based index
    return ranked_pages
Define the path to your PDF and perform a hybrid search with a query:

python
Copy code
pdf_path = '/content/DL BOOK .pdf'  # Replace with your actual PDF path
query = "Single Computational Layer: The Perceptron"
top_k = 1  # Number of top results to retrieve
results = hybrid_search_in_pdf(pdf_path, query, top_k=top_k)
if results:
    page_number, page_text = results[0]
    print(f"Most Relevant Page: {page_number}")
    print(f"Answer Text: {page_text}")
else:
    print("No relevant page found.")
