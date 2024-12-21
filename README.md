# Comprehensive README for Generative AI Engineering Assessment Tasks

## Table of Contents
- [Overview](#overview)
- [Tasks](#tasks)
  - [Task 1: Book Summarization](#task-1-book-summarization)
  - [Task 2: Personalized Student Study Plan Generator](#task-2-personalized-student-study-plan-generator)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributions](#contributions)
- [License](#license)
- [Author](#author)

## Overview
This repository contains assessment tasks designed for a Generative AI Engineer role, showcasing the ability to leverage advanced AI techniques for text processing, summarization, and personalized planning. The tasks utilize LangChain and OpenAI's GPT models to deliver intelligent solutions for summarizing literature and creating tailored academic plans.

## Tasks

### Task 1: Book Summarization
**Objective:** Generate a concise summary of Fyodor Dostoevsky's "Crime and Punishment," limited to 20 pages.

**Key Features:**
- **Text Loading and Cleaning:** Utilizes `PyPDFLoader` to load and preprocess the text from the PDF.
- **Semantic Chunking:** Implements `SemanticChunker` for intelligent segmentation of the text into meaningful chunks.
- **Embedding Generation:** Uses OpenAI embeddings to capture the semantic meaning of the text.
- **Clustering:** Applies FAISS for clustering the text chunks to enhance summarization.
- **PDF Generation:** Outputs the final summary into a well-structured PDF format using `FPDF`.

**Code Explanation:**

```python
# Import necessary libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
import pandas as pd
import numpy as np
import faiss
import re
import os
```
Library Imports: The code begins by importing necessary libraries for document loading, text processing, embeddings, and data manipulation.


```python
# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your API key"
```
API Key Configuration: The OpenAI API key is set in the environment to allow access to the OpenAI services.

```python
# Load the book
loader = PyPDFLoader("crime-and-punishment.pdf")
pages = loader.load_and_split()
```
Loading the PDF: The PyPDFLoader is used to load the PDF file "crime-and-punishment.pdf" and split it into individual pages for processing.


```python
# Clean the text
def clean_text(text):
    cleaned_text = text
    cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Remove extra spaces
    cleaned_text = cleaned_text.replace('\n', ' ')  # Replace newline characters with spaces
    return cleaned_text

```
Text Cleaning Function: A function clean_text is defined to remove extra spaces and newline characters from the text, ensuring a cleaner input for further processing.

```python
# Combine pages and clean the text
pages = pages[7:]  # Cut out the open and closing parts
text = ' '.join([page.page_content.replace('\t', '') for page in pages])
cleaned_text = clean_text(text)
```

Combining and Cleaning Pages: The pages are combined into a single string, and the cleaning function is applied to prepare the text for analysis.

```python
# Initialize OpenAI model
llm = OpenAI()

```
Model Initialization: An instance of the OpenAI model is created to facilitate token counting and further processing.

```python
# Get the number of tokens in the book
tokens = llm.get_num_tokens(cleaned_text)
print(f"We have {tokens} tokens in the book")

```
Token Counting: The total number of tokens in the cleaned text is calculated and printed, which is important for understanding the text's length and complexity.

```python
# Split the text into semantic chunks
text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="interquartile")
docs = text_splitter.create_documents([cleaned_text])
```
Semantic Chunking: The text is split into semantic chunks using SemanticChunker, which helps in organizing the text into manageable pieces for summarization.


```python
# Generate embeddings for the documents
embeddings = get_embeddings([doc.page_content for doc in docs])
```
Embedding Generation: The embeddings for each document chunk are generated using a function get_embeddings, which captures the semantic meaning of the text.
