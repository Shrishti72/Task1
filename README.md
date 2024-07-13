# Hybrid PDF Search System

This repository contains three Python scripts for performing text search in a PDF file using different methods: semantic search, TF-IDF-based textual search, and a hybrid approach combining both methods. Each script extracts text from all pages of a PDF file and searches for the most relevant page based on a given query.

## Table of Contents
- [Dependencies](#dependencies)
- [Semantic Search](#semantic-search)
- [Textual Search with TF-IDF](#textual-search-with-tfidf)
- [Hybrid Search](#hybrid-search)

## Semantic Search
This script utilizes a pre-trained transformer model to perform semantic search in a PDF. Semantic search focuses on the meaning and context of words rather than exact matches.

## Textual Search with TF-IDF
The TF-IDF (Term Frequency-Inverse Document Frequency) approach is used in this script for textual search within a PDF. TF-IDF calculates the importance of a word in a document relative to its frequency in other documents.

## Hybrid Search
The hybrid search script combines both semantic search using a pre-trained transformer model and textual search using TF-IDF. This approach aims to leverage both semantic understanding and textual relevance for more accurate search results.

Each script requires specifying the path to the PDF file and a query for which the most relevant page is retrieved along with the corresponding text.
