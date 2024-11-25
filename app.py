import os
import streamlit as st
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set your Groq API Key (use environment variable for security)
GROQ_API_KEY = "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"  # Ensure this environment variable is set
client = Groq(api_key=GROQ_API_KEY)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths to your assets folder
assets_folder = os.path.join(os.getcwd(), 'assets')

# Function to load resources from local storage
def load_resources():
    # Paths to index and chunk files
    industry_index_path = os.path.join( 'industry_index.faiss')
    industry_chunks_path = os.path.join( 'industry_chunks.pkl')
    circular_index_path = os.path.join( 'circular_index.faiss')
    circular_chunks_path = os.path.join( 'circular_chunks.pkl')

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

# Prepare data
industry_index, industry_chunks, circular_index, circular_chunks = load_resources()

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
            relevant_chunks = retrieve_relevant_chunks(user_query, circular_index, circular_chunks)
            context = "\n".join(relevant_chunks)
            prompt =  f"""
You are an expert RBI compliance analyst. Based on the provided RBI Master Circular on Management of Advances:

{context}

Please analyze the following scenario for compliance:
{user_query}

Provide a detailed compliance analysis with the following structure:

1. Compliance Status:
- Clear statement whether the scenario is compliant or non-compliant
- Level of certainty in the assessment

2. Relevant Circular Details:
- Specific section(s) and paragraph references
- Direct quotes from applicable sections where relevant

3. Detailed Analysis:
- Breakdown of key compliance requirements
- Calculation/numerical analysis if applicable
- Specific points of compliance/non-compliance

4. Additional Considerations:
- Related requirements or obligations
- Monitoring/reporting requirements if applicable

5. Recommendation:
- Clear guidance on what needs to be done for compliance
- Specific steps to address any non-compliance

Please provide definitive guidance based solely on the circular content, avoiding ambiguity or speculation.

Response:
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

# Function for Industry Classification (Problem Statement 3)
def industry_classification():
    st.header("Industry Classification Assistant")
    user_keywords = st.text_input("Enter keywords related to the industry:", key='industry_input')
    if st.button("Get Industry Classification", key='industry_button'):
        if user_keywords:
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

    user_input = st.text_input("You:", key="model1_input")

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
                    temperature=0.0
                )
                response = chat_completion.choices[0].message.content.strip()
                st.session_state.chat_history.append(("Model", response))
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your API key and model availability.")

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Model 1:** {message}")


def retrieve_relevant_financial_statements(query, index, statements, model, top_k=10, max_tokens=1500):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    retrieved_statements = []
    total_tokens = 0
    for idx in indices[0]:
        statement = statements[idx]['statement']
        token_count = len(statement.split())
        if total_tokens + token_count > max_tokens:
            break
        retrieved_statements.append(statements[idx])
        total_tokens += token_count
    return retrieved_statements


def model2_financial_data():
    st.header("Financial Data Assistant (Model 2)")

    # Load the FAISS index and financial statements
    financial_index_path = os.path.join( 'financial_index.faiss')
    financial_statements_path = os.path.join( 'financial_statements.pkl')

    # Load FAISS index
    if not os.path.exists(financial_index_path):
        st.error("Financial FAISS index not found.")
        st.stop()
    financial_index = faiss.read_index(financial_index_path)

    # Load statements
    if not os.path.exists(financial_statements_path):
        st.error("Financial statements data not found.")
        st.stop()
    with open(financial_statements_path, 'rb') as f:
        financial_statements = pickle.load(f)

    # Allow the user to input a query
    user_query = st.text_area("Ask a question about Indian state-wise financial details (1980-2015):", key='model2_input')

    if st.button("Get Answer", key='model2_button'):
        if user_query:
            # Extract metric, state, and year from the user's query
            import re

            # List of possible metrics
            metrics_list = [
                'aggregate expenditure', 'capital expenditure', 'gross fiscal deficits',
                'nominal gsdp series', 'own tax revenues', 'revenue deficits',
                'revenue expenditure', 'social sector expenditure'
            ]

            # Create a pattern to match any of the metrics
            metrics_pattern = '|'.join(metrics_list)
            metric_regex = re.compile(rf'\b({metrics_pattern})\b', re.IGNORECASE)

            # Extract metric
            metric_match = metric_regex.search(user_query)
            if metric_match:
                query_metric = metric_match.group(1).strip().title()
            else:
                query_metric = None

            # Extract state
            # Assuming state names are capitalized properly in the data
            states_list = list(set(s['state'] for s in financial_statements))
            states_pattern = '|'.join(states_list)
            state_regex = re.compile(rf'\b({states_pattern})\b', re.IGNORECASE)
            state_match = state_regex.search(user_query)
            if state_match:
                query_state = state_match.group(1).strip()
            else:
                query_state = None

            # Extract year
            year_regex = re.compile(r'(\d{4}(?:-\d{2})?)')
            year_match = year_regex.search(user_query)
            if year_match:
                query_year = year_match.group(1)
                # Normalize the year format if needed
                if len(query_year) == 4:
                    # Convert "1992" to "1992-93"
                    query_year = f"{query_year}-{str(int(query_year[-2:])+1).zfill(2)}"
                elif len(query_year) == 7:
                    # Already in "1992-93" format
                    pass
            else:
                query_year = None

            if query_state and query_year:
                # Collect data based on the extracted information
                data = {}
                for s in financial_statements:
                    if (
                        s['state'].lower() == query_state.lower() and
                        s['year'] == query_year
                    ):
                        if query_metric:
                            if s['metric_type'].lower() == query_metric.lower():
                                data[s['metric_type']] = s['value']
                                break  # Since we found the specific metric, we can stop
                        else:
                            data[s['metric_type']] = s['value']

                if data:
                    if query_metric:
                        # Display only the specific metric
                        value = data.get(query_metric)
                        if value is not None:
                            st.write(f"The {query_metric} of {query_state} in {query_year} is {value}")
                        else:
                            st.write(f"{query_metric} data not found for {query_state} in {query_year}.")
                    else:
                        # Display all metrics
                        st.write(f"Financial data for **{query_state}** in **{query_year}**:")
                        df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                        st.table(df)
                else:
                    st.write("Data not found for the specified state, year, or metric.")
            else:
                st.write("Could not understand the query. Please specify the state and year.")

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
