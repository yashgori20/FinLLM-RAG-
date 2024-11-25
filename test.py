import os
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set your Groq API Key (ensure this environment variable is set)
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths to your assets folder
assets_folder = os.path.join(os.getcwd(), 'assets')

# Function to load resources for industry and circular compliance
def load_resources():
    # Paths to index and chunk files
    industry_index_path = os.path.join(assets_folder, 'industry_index.faiss')
    industry_chunks_path = os.path.join(assets_folder, 'industry_chunks.pkl')
    circular_index_path = os.path.join(assets_folder, 'circular_index.faiss')
    circular_chunks_path = os.path.join(assets_folder, 'circular_chunks.pkl')

    # Check if the files exist
    if not all(os.path.exists(path) for path in [industry_index_path, industry_chunks_path, circular_index_path, circular_chunks_path]):
        st.error("FAISS indexes and chunk files not found in the assets folder. Please ensure they are present.")
        st.stop()

    # Load FAISS indexes and chunks
    industry_index = faiss.read_index(industry_index_path)
    with open(industry_chunks_path, 'rb') as f:
        industry_chunks = pickle.load(f)
    circular_index = faiss.read_index(circular_index_path)
    with open(circular_chunks_path, 'rb') as f:
        circular_chunks = pickle.load(f)
    return industry_index, industry_chunks, circular_index, circular_chunks

# Function to load resources for Model 2 (financial data)
def load_financial_data_resources():
    # Paths to index and data texts
    data_index_path = os.path.join(assets_folder, 'financial_data_index.faiss')
    data_texts_path = os.path.join(assets_folder, 'financial_data_texts.pkl')

    # Check if the files exist
    if not all(os.path.exists(path) for path in [data_index_path, data_texts_path]):
        st.error("Financial data index and texts not found in the assets folder. Please ensure they are present.")
        st.stop()

    # Load FAISS index and data texts
    data_index = faiss.read_index(data_index_path)
    with open(data_texts_path, 'rb') as f:
        data_texts = pickle.load(f)
    return data_index, data_texts

# Prepare data
industry_index, industry_chunks, circular_index, circular_chunks = load_resources()
data_index, data_texts = load_financial_data_resources()

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# Function for Circular Compliance (Problem Statement 2)
def circular_compliance():
    st.header("Circular Compliance Assistant")
    user_query = st.text_area("Enter your scenario or question:", key='circular_input')
    if st.button("Check Compliance", key='circular_button'):
        if user_query:
            try:
                relevant_chunks = retrieve_relevant_chunks(user_query, circular_index, circular_chunks, top_k=10)
                context = "\n".join(relevant_chunks)
                prompt = f"""
You are an expert assistant helping to check compliance with RBI Master Circulars. Based on the following excerpts:

{context}

User's Scenario or Question:
{user_query}

Provide a detailed and precise analysis indicating whether the scenario is compliant with the RBI circulars. Your response should include:

- **Direct Answer**: Clearly state whether the action is permitted or prohibited.
- **Specific References**: Cite relevant circular numbers and specific sections.
- **Key Compliance Points**: Highlight important points from the circular that apply to the scenario.
- **Explanation**: Briefly explain why the action is compliant or not.
- **Recommendation**: Suggest what the bank should do in this situation.

Please format your response with headings for each section (e.g., "Direct Answer", "Specific References", etc.).

Answer:
"""
                chat_completion = client.chat.completions.create(
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    model="gemma2-9b-it",
                    stream=False,
                    temperature=0.0
                )
                response = chat_completion.choices[0].message.content.strip()
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and model availability.")
        else:
            st.info("Please enter a scenario or question to proceed.")

# Function for Industry Classification (Problem Statement 3)
def industry_classification():
    st.header("Industry Classification Assistant")
    user_keywords = st.text_input("Enter keywords related to the industry:", key='industry_input')
    if st.button("Get Industry Classification", key='industry_button'):
        if user_keywords:
            try:
                relevant_chunks = retrieve_relevant_chunks(user_keywords, industry_index, industry_chunks)
                context = "\n".join(relevant_chunks)
                prompt = f"""
You are an assistant helping to classify industries based on keywords. Based on the following information:

{context}

User's Keywords:
{user_keywords}

Suggest the most appropriate industry classification codes. Ask any necessary follow-up questions to clarify if needed.

Answer:
"""
                chat_completion = client.chat.completions.create(
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    model="gemma2-9b-it",
                    stream=False,
                    temperature=0.0
                )
                response = chat_completion.choices[0].message.content.strip()
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and model availability.")
        else:
            st.info("Please enter keywords to proceed.")

# Existing calculation function (Problem Statement 1)
def calculations():
    st.subheader("Calculation Methodology")
    calc_option = st.selectbox("Choose Calculation Method",
                             ("Maximum Permissible Bank Finance (MPBF)", "Drawing Power (DP)"))

    if calc_option == "Maximum Permissible Bank Finance (MPBF)":
        st.header("MPBF Calculation")
        total_current_assets = st.number_input("Total Current Assets (TCA):", min_value=0.0, value=0.0)
        other_current_liabilities = st.number_input("Other Current Liabilities (OCL):", min_value=0.0, value=0.0)
        actual_nwc = st.number_input("Actual/Projected Net Working Capital (NWC):", min_value=0.0, value=0.0)

        if st.button("Calculate MPBF"):
            working_capital_gap = total_current_assets - other_current_liabilities
            minimum_stipulated_nwc = 0.25 * total_current_assets
            item_6 = working_capital_gap - minimum_stipulated_nwc
            item_7 = working_capital_gap - actual_nwc
            mpbf = min(item_6, item_7)

            st.success(f"Working Capital Gap (WCG): {working_capital_gap:.2f}")
            st.success(f"Minimum Stipulated NWC (25% of TCA): {minimum_stipulated_nwc:.2f}")
            st.success(f"Item 6 (WCG - Minimum Stipulated NWC): {item_6:.2f}")
            st.success(f"Item 7 (WCG - Actual NWC): {item_7:.2f}")
            st.success(f"Maximum Permissible Bank Finance (MPBF): {mpbf:.2f}")

    elif calc_option == "Drawing Power (DP)":
        st.header("DP Calculation")
        inventory_margin = 0.25
        receivables_margin = 0.40
        creditors_margin = 0.40

        st.subheader("Inventory Details")
        raw_material = st.number_input("Raw Material:", min_value=0.0, value=0.0)
        consumable_spares = st.number_input("Other Consumable Spares:", min_value=0.0, value=0.0)
        stock_in_process = st.number_input("Stock-in-process:", min_value=0.0, value=0.0)
        finished_goods = st.number_input("Finished Goods:", min_value=0.0, value=0.0)

        st.subheader("Receivables")
        domestic_receivables = st.number_input("Domestic Receivables:", min_value=0.0, value=0.0)
        export_receivables = st.number_input("Export Receivables:", min_value=0.0, value=0.0)

        st.subheader("Creditors")
        creditors = st.number_input("Creditors:", min_value=0.0, value=0.0)

        if st.button("Calculate DP"):
            inventory_total = raw_material + consumable_spares + stock_in_process + finished_goods
            inventory_advance = inventory_total * (1 - inventory_margin)
            receivables_total = domestic_receivables + export_receivables
            receivables_advance = receivables_total * (1 - receivables_margin)
            creditors_advance = creditors * (1 - creditors_margin)
            total_A = inventory_advance + receivables_advance
            total_B = creditors_advance
            dp = total_A - total_B

            st.success(f"Total Inventory (After Margin): {inventory_advance:.2f}")
            st.success(f"Total Receivables (After Margin): {receivables_advance:.2f}")
            st.success(f"Total (A): {total_A:.2f}")
            st.success(f"Creditors (After Margin): {total_B:.2f}")
            st.success(f"Drawing Power (DP): {dp:.2f}")

# Function for Model 1 chat interface
def run_model1_chat():
    st.header("Model 1 Chat Interface")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'user_input_model1' not in st.session_state:
        st.session_state['user_input_model1'] = ''

    user_input = st.text_input("You:", value=st.session_state['user_input_model1'], key="model1_input")

    if st.button("Send", key='model1_send'):
        if user_input:
            st.session_state.chat_history.append(("User", user_input))

            try:
                # Get model response
                chat_completion = client.chat.completions.create(
                    messages=[
                        {'role': 'user', 'content': user_input}
                    ],
                    model="gemma2-9b-it",
                    stream=False,
                    temperature=0.0  # Set temperature to zero as per your request
                )
                response = chat_completion.choices[0].message.content.strip()
                st.session_state.chat_history.append(("Model", response))
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and model availability.")

            # Clear the input
            st.session_state['user_input_model1'] = ''
            st.experimental_rerun()

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Model 1:** {message}")

# Function for Model 2 (Financial Data Assistant)
def model2_financial_data():
    st.header("Financial Data Assistant (Model 2)")

    # Load resources
    data_index, data_texts = load_financial_data_resources()

    # User input
    user_query = st.text_area("Ask a question about Indian state-wise financial details (1980-2015):", key='model2_input')

    if st.button("Get Answer", key='model2_button'):
        if user_query:
            try:
                # Retrieve relevant data chunks
                query_embedding = model.encode([user_query], convert_to_numpy=True)
                distances, indices = data_index.search(query_embedding, k=5)
                retrieved_texts = [data_texts[i] for i in indices[0]]

                # Prepare the context
                context = "\n".join(retrieved_texts)

                prompt = f"""
You are an expert assistant helping to answer questions about Indian state-wise financial details from 1980 to 2015. Based on the following data:

{context}

User's Question:
{user_query}

Provide a clear and accurate answer based on the data provided. If the data is insufficient to answer the question, inform the user accordingly.

Answer:
"""
                chat_completion = client.chat.completions.create(
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    model="gemma2-9b-it",
                    stream=False,
                    temperature=0.0  # Set temperature to zero as per your request
                )
                response = chat_completion.choices[0].message.content.strip()
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and model availability.")
        else:
            st.info("Please enter a question to proceed.")

# Main function to run the app
def main():
    st.set_page_config(page_title="Finance Assistant", page_icon="💸", layout="wide")
    st.title("💸 Finance Assistant")

    option = st.radio(
        "Choose a Functionality",
        ("Calculation Methodology", "Circular Compliance", "Industry Classification", "Model 1", "Model 2")
    )

    if option == "Calculation Methodology":
        calculations()
    elif option == "Circular Compliance":
        circular_compliance()
    elif option == "Industry Classification":
        industry_classification()
    elif option == "Model 1":
        run_model1_chat()
    elif option == "Model 2":
        model2_financial_data()

if __name__ == "__main__":
    main()
