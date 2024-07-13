# Hybrid PDF Search System

This repository contains three Python scripts for performing text search in a PDF file using different methods: semantic search, TF-IDF-based textual search, and a hybrid approach combining both methods. Each script extracts text from all pages of a PDF file and searches for the most relevant page based on a given query.

## Table of Contents
- [Dependencies](#dependencies)
- [Semantic Search](#semantic-search)
- [Textual Search with TF-IDF](#textual-search-with-tfidf)
- [Hybrid Search](#hybrid-search)

## Dependencies

Make sure to install the following dependencies before running the scripts:

```bash
pip install PyMuPDF
pip install torch
pip install transformers
pip install scikit-learn


Semantic Search
This script uses a pre-trained transformer model to embed the text and perform a semantic search based on cosine similarity.

Usage
Import necessary libraries:
```bash
import fitz 
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
