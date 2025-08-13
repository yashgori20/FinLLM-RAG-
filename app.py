import os
import streamlit as st
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

@st.cache_resource
def load_sentence_transformer():
    try:
        import torch
        # Force CPU to avoid meta tensor issues
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model = model.to('cpu')  # Explicitly move to CPU
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        try:
            # Try alternative initialization without device specification
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            st.error("Please reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            st.stop()

assets_folder = os.path.join(os.getcwd(), 'assets')

def load_resources():
    industry_index_path = os.path.join( 'industry_index.faiss')
    industry_chunks_path = os.path.join( 'industry_chunks.pkl')
    circular_index_path = os.path.join( 'circular_index.faiss')
    circular_chunks_path = os.path.join( 'circular_chunks.pkl')
    if not all(os.path.exists(path) for path in [industry_index_path, industry_chunks_path, circular_index_path, circular_chunks_path]):
        st.error("FAISS indexes and chunk files not found in the assets folder. Please ensure they are present.")
        st.stop()
    industry_index = faiss.read_index(industry_index_path)
    with open(industry_chunks_path, 'rb') as f:
        industry_chunks = pickle.load(f)
    circular_index = faiss.read_index(circular_index_path)
    with open(circular_chunks_path, 'rb') as f:
        circular_chunks = pickle.load(f)
    return industry_index, industry_chunks, circular_index, circular_chunks
industry_index, industry_chunks, circular_index, circular_chunks = load_resources()

def retrieve_relevant_chunks(query, index, chunks, top_k=10):
    model = load_sentence_transformer()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Get more chunks initially and filter for relevance
    retrieved_chunks = []
    query_lower = query.lower()
    
    # Check if query is about general term loans vs share financing
    is_general_loan_query = any(term in query_lower for term in [
        'term loan', 'manufacturing', 'documentation requirement', 
        'credit sanction', 'loan sanction', 'general lending'
    ]) and not any(term in query_lower for term in [
        'share', 'debenture', 'bond', 'equity', 'capital market'
    ])
    
    for i, idx in enumerate(indices[0]):
        chunk_text = str(chunks[idx]).lower()
        
        # If it's a general loan query, deprioritize share-related chunks
        if is_general_loan_query and any(term in chunk_text for term in [
            'advances against shares', 'debentures', 'bonds', 'capital market',
            'shareholding', 'equity acquisition'
        ]):
            # Skip clearly irrelevant share-related chunks for general loan queries
            continue
            
        retrieved_chunks.append(chunks[idx])
        if len(retrieved_chunks) >= 5:  # Return top 5 relevant chunks
            break
    
    # If we don't have enough chunks, add some of the skipped ones
    if len(retrieved_chunks) < 3:
        for idx in indices[0]:
            if len(retrieved_chunks) >= 5:
                break
            if chunks[idx] not in retrieved_chunks:
                retrieved_chunks.append(chunks[idx])
    
    return retrieved_chunks
    
def circular_compliance():
    st.header("Circular Compliance Assistant")
    st.markdown("**Example scenarios you can ask about:**")
    st.markdown("• *A bank is providing working capital finance to a textile company. The company's current assets are ₹100 crores and current liabilities are ₹60 crores. Is the bank compliant with MPBF norms if they provide ₹35 crores as working capital finance?*")
    st.markdown("• *What are the documentation requirements for sanctioning term loans above ₹5 crores to manufacturing companies?*")
    st.markdown("• *Can a bank provide additional working capital finance if the borrower's drawing power calculation shows negative figures?*")
    user_query = st.text_area("Enter your scenario or question:", key='circular_input')
    if st.button("Check Compliance", key='circular_button'):
        if user_query:
            relevant_chunks = retrieve_relevant_chunks(user_query, circular_index, circular_chunks)
            context = "\n".join(relevant_chunks)
            prompt = f"""
You are an expert RBI compliance analyst. Based on the provided RBI Master Circular on Management of Advances:

{context}

Please analyze the following scenario for compliance:
{user_query}

CRITICAL INSTRUCTIONS:
- If the provided context is about share financing, debentures, bonds, or capital market exposures, and the query is about GENERAL TERM LOANS, clearly state that the retrieved information is not relevant to the query
- Focus ONLY on requirements that apply to standard term loans to manufacturing/business entities  
- Do NOT conflate share financing requirements with general term loan requirements
- If the context doesn't contain information relevant to the specific query, state this clearly and indicate what type of information would be needed

Provide analysis with this structure:
1. Relevance Assessment: Is the provided context relevant to the query?
2. Actual Requirements: What are the real requirements for this scenario based on relevant sections?
3. Documentation: Specific documents actually required
4. Approval Process: Required approvals and delegation levels
5. Compliance Steps: Practical steps for compliance

Base your response ONLY on information directly relevant to the query type.
Response:
"""
            chat_completion = client.chat.completions.create(
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                model="openai/gpt-oss-120b",
                stream=False,
                temperature=0.0
            )
            response = chat_completion.choices[0].message.content.strip()
            st.write(response)

def industry_classification():
    st.header("Industry Classification Assistant")
    st.markdown("**Example keywords you can search for:**")
    st.markdown("• *textile manufacturing, cotton spinning, garments*")
    st.markdown("• *software development, IT services, application development*")
    st.markdown("• *food processing, dairy products, beverages*")
    st.markdown("• *automobile parts, automotive components, vehicle manufacturing*")
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
                model="openai/gpt-oss-120b",
                stream=False,
                temperature=0.0
            )
            response = chat_completion.choices[0].message.content.strip()
            st.write(response)

def calculations():
    st.subheader("Calculation Methodology")
    st.markdown("**Available Calculations:**")
    st.markdown("• **MPBF (Maximum Permissible Bank Finance)**: Calculate the maximum working capital finance a bank can provide based on RBI norms")
    st.markdown("• **Drawing Power (DP)**: Calculate the borrowing limit based on current assets with applicable margins")
    calc_option = st.selectbox("Choose Calculation Method",
                             ("Maximum Permissible Bank Finance (MPBF)", "Drawing Power (DP)"))
    if calc_option == "Maximum Permissible Bank Finance (MPBF)":
        st.header("MPBF Calculation")
        st.markdown("**Example:** TCA: ₹100 crores, OCL: ₹30 crores, Actual NWC: ₹20 crores")
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
        st.markdown("**Example:** Raw Material: ₹20 crores, Finished Goods: ₹15 crores, Receivables: ₹25 crores, Creditors: ₹10 crores")
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
def main():
    st.set_page_config(page_title="Finance Assistant", page_icon="💸", layout="wide")
    st.title("💸 Finance Assistant")
    option = st.radio(
        "Choose a Functionality",
        ("Calculation Methodology", "Circular Compliance", "Industry Classification")
    )
    if option == "Calculation Methodology":
        calculations()
    elif option == "Circular Compliance":
        circular_compliance()
    elif option == "Industry Classification":
        industry_classification()
if __name__ == "__main__":
    main()
