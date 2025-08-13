---
title: FinLLM RAG
emoji: ⚡
colorFrom: purple
colorTo: gray
sdk: streamlit
sdk_version: 1.40.1
app_file: app.py
pinned: false
---

# 💸 Finance Assistant

This project is a multi-functional financial assistant built with Streamlit. It leverages large language models and retrieval-augmented generation (RAG) to provide a suite of tools for financial analysis, compliance, and data retrieval.

## Features

The application is divided into several key functionalities:

  * **Circular Compliance Assistant**: Analyzes user-provided scenarios for compliance against RBI Master Circulars on Management of Advances. It uses a FAISS vector database to retrieve relevant sections of the circular and a language model to generate a detailed compliance report.
  * **Industry Classification Assistant**: Suggests appropriate industry classification codes based on user-provided keywords. This feature also utilizes a RAG pipeline to search through an industry classification master document.
  * **Calculation Methodology**: Provides interactive calculators for key financial metrics:
      * **Maximum Permissible Bank Finance (MPBF)**
      * **Drawing Power (DP)**
  * **Financial Data Assistant**: Answers questions about historical (1980-2015) state-wise financial data for India. It can retrieve specific metrics for a given state and year.
  * **Model 1 Chat**: A general-purpose chat interface powered by the `gemma2-9b-it` model via the Groq API.

## How It Works

The core of the "Circular Compliance" and "Industry Classification" assistants is a Retrieval-Augmented Generation (RAG) pipeline.

1.  **Indexing**: Source documents (`Master Circular.pdf`, `Industry Classification Master.pdf`) are chunked, and the text chunks are converted into vector embeddings using a `SentenceTransformer` model. These embeddings are stored in a FAISS index for efficient similarity search.
2.  **Retrieval**: When a user enters a query, the query is embedded, and the FAISS index is searched to find the most relevant document chunks.
3.  **Generation**: The retrieved chunks are passed as context, along with the user's query, to a large language model (`gemma2-9b-it`). The model then generates a comprehensive and context-aware response.

The "Financial Data Assistant" works by directly parsing the user's query for state, year, and metric information and looking up the corresponding data from a pre-loaded data file.

## Setup and Installation

1.  **Clone the Repository**:

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install Dependencies**:
    Install the necessary Python libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Assets**:
    The application requires pre-built FAISS indexes and data files.

      * Create a folder named `assets` in the root directory.
      * Generate and place the following files into the `assets` folder (You will need a separate script to process the source PDFs and JSON to create these files):
          * `industry_index.faiss`
          * `industry_chunks.pkl`
          * `circular_index.faiss`
          * `circular_chunks.pkl`
          * `financial_index.faiss`
          * `financial_statements.pkl`

4.  **API Key**:
    Insert your Groq API key directly into the `app.py` file at the following line:

    ```python
    GROQ_API_KEY = "your-groq-api-key-here"
    ```

## Usage

1.  **Run the Streamlit App**:
    Execute the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the Application**:

      * Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
      * Use the radio buttons at the top of the page to navigate between the different functionalities: "Calculation Methodology", "Circular Compliance", "Industry Classification", "Model 1", and "Model 2".
      * Follow the on-screen instructions for each tool.

## Dependencies

This project relies on the following major libraries:

  * `streamlit`: For creating the web application interface.
  * `groq`: The client for accessing the Groq API.
  * `sentence-transformers`: For generating text embeddings.
  * `faiss-cpu`: For efficient similarity search in the vector database.
  * `pandas`: For data manipulation, particularly in the financial data assistant.
  * `numpy`: For numerical operations.
  * `torch` & `transformers`: Core dependencies for the sentence transformer models.