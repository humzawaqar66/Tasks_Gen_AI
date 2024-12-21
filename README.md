# README for Generative AI Engineering Assessment Tasks

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


### Task 2: Personalized Study Plans For Students Using LangChain

## Overview
This task involves creating a personalized study plan generator for students using LangChain and OpenAI's GPT model. The generator takes into account various factors such as the student's name, field of study, year of study, subjects, learning styles, personal objectives, challenges, and extracurricular activities to create a tailored study plan.

## Key Features
- **Personalized Study Plans:** Generates study plans customized to the individual needs and preferences of students.
- **Integration with OpenAI:** Utilizes OpenAI's GPT model to generate detailed and actionable study plans.
- **Flexible Input:** Accepts various input parameters to cater to different student profiles.

## Code Explanation

```python
# Import necessary libraries
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
```
Library Imports: The code begins by importing the necessary libraries from LangChain. PromptTemplate is used to create structured prompts, and ChatOpenAI is used to interact with the OpenAI model. The os library is imported to manage environment variables.

```python
# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your API key"
```
API Key Configuration: The OpenAI API key is set in the environment to allow access to the OpenAI services. Replace "your API key" with your actual OpenAI API key.

```python
# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
```
Model Initialization: An instance of the OpenAI model is created with a specified temperature. The temperature controls the randomness of the model's responses; a lower value results in more deterministic outputs.

```python
# Define the detailed prompt template
prompt_template = PromptTemplate(
    input_variables=[
        "name",
        "field_of_study",
        "year_of_study",
        "list_of_subjects",
        "preferred_learning_styles",
        "personal_objectives",
        "challenges",
        "extracurricular_activities"
    ],
    template=(
        "You are an expert academic planner. Your task is to create a personalized study plan for a student based on the following information:\n"
        "Student Name: {name}\n"
        "Field of Study: {field_of_study}\n"
        "Year of Study: {year_of_study}\n"
        "List of Subjects: {list_of_subjects} (e.g., Math, Physics, Literature)\n"
        "Preferred Learning Styles: {preferred_learning_styles} (e.g., visual, auditory, kinesthetic)\n"
        "Personal Objectives: {personal_objectives} (e.g., preparing for exams, improving specific subjects)\n"
        "Challenges: {challenges} (e.g., time management issues, difficulty concentrating)\n"
        "Extracurricular Activities: {extracurricular_activities} (e.g., sports, music, coding)\n\n"
        "Using this information, create a study plan that includes:\n"
        "1. Weekly schedules customized to the student's learning style and academic requirements.\n"
        "2. Strategies to address the challenges mentioned.\n"
        "3. Suggestions for balancing extracurricular activities with academic goals.\n"
        "4. Specific action items for achieving personal objectives.\n"
        "5. Resources or tools that align with the student's learning style and needs.\n\n"
        "Ensure the plan is detailed, actionable, and motivating for the student."
    )
)

```
Prompt Template Definition: A detailed prompt template is defined to guide the model in generating a personalized study plan. The input_variables list specifies the parameters that will be filled in the template, and the template string outlines the structure and content of the prompt.

```python
# Example usage
example_input = {
    "name": "Alex Johnson",
    "field_of_study": "Computer Science",
    "year_of_study": "2nd Year",
    "list_of_subjects": "Data Structures, Algorithms, Operating Systems",
    "preferred_learning_styles": "Visual and kinesthetic",
    "personal_objectives": "Excel in Data Structures and prepare for internships",
    "challenges": "Balancing coursework with personal projects",
    "extracurricular_activities": "Robotics Club, Basketball"
}
```
Example Input: An example input dictionary is created to simulate a student's information. This input will be used to fill the prompt template.

```python
# Render the prompt with example data
filled_prompt = prompt_template.format(**example_input)
```
Prompt Rendering: The prompt is filled with the example data using the format method, which replaces the placeholders in the template with actual values

## Installation

To set up the environment for the Personalized Study Plans generator, you need to install the required packages. You can do this using pip. Run the following commands in your terminal:

```bash
pip install langchain
pip install langchain_openai
pip install pypdf
pip install faiss-cpu
pip install openai
```

## Usage

To use the Personalized Study Plans generator, follow these steps:

1. **Set Up Your Environment:**
   Ensure that you have installed all the required packages as mentioned in the Installation section.

2. **Obtain Your OpenAI API Key:**
   Sign up for an OpenAI account and obtain your API key. Replace `"your API key"` in the code with your actual OpenAI API key.

3. **Run the Code:**
   You can run the provided code in a Python environment (e.g., Jupyter Notebook, Google Colab, or any Python IDE). 

4. **Input Student Information:**
   Modify the `example_input` dictionary in the code to include the relevant information for the student for whom you want to create a personalized study plan. The fields include:
   - `name`: The student's name.
   - `field_of_study`: The student's field of study (e.g., Computer Science).
   - `year_of_study`: The current year of study (e.g., 2nd Year).
   - `list_of_subjects`: A comma-separated list of subjects the student is taking.
   - `preferred_learning_styles`: The student's preferred learning styles (e.g., Visual and kinesthetic).
   - `personal_objectives`: The student's personal objectives (e.g., Excel in Data Structures).
   - `challenges`: Any challenges the student faces (e.g., time management issues).
   - `extracurricular_activities`: A list of extracurricular activities the student is involved in (e.g., Robotics Club, Basketball).

5. **Generate the Study Plan:**
   After modifying the input, run the code to generate the personalized study plan. The output will be printed to the console.

6. **Review the Output:**
   Review the generated study plan to ensure it meets the student's needs and preferences. You can further customize the plan as necessary.

By following these steps, you can effectively use the Personalized Study Plans generator to create tailored academic plans for students.

## Requirements

To successfully run the Personalized Study Plans generator, ensure that your environment meets the following requirements:

- **Python Version:** Python 3.7 or higher is recommended.
- **Pip:** Ensure that pip is installed for package management.
- **OpenAI API Key:** You must have a valid OpenAI API key to access the GPT model.
- **Required Packages:** The following packages must be installed:
  - `langchain`: For building the prompt and managing interactions with the OpenAI model.
  - `langchain_openai`: For integrating OpenAI's capabilities within LangChain.
  - `pypdf`: For handling PDF documents if needed in other tasks.
  - `faiss-cpu`: For efficient similarity search and clustering of embeddings.
  - `openai`: For accessing OpenAI's API.

Make sure to install the required packages using the commands provided in the Installation section before running the code.

## Contributors

This project was developed by Humza Waqar as part of the assessment for the Generative AI Engineer position at Cogent Labs. Contributions, feedback, and suggestions are welcome. If you would like to contribute to this project, please feel free to reach out.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you include the original license and copyright notice in any copies or substantial portions of the software.

For more details, please refer to the LICENSE file included in this repository.
