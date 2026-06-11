import streamlit as st
import pandas as pd
import json
import tempfile
import requests
import os
from io import BytesIO
from typing import Dict, Any, List

# File extraction libraries
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

# ------------------------------------------------------------------
# 1. Custom AnythingLLM Client
# ------------------------------------------------------------------
class ChatAnythingLLM:
    def __init__(self, base_url: str, api_key: str, workspace_slug: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.workspace_slug = workspace_slug

    def invoke(self, messages):
        """
        Send a message and get a response from AnythingLLM.
        Returns a response object with a .content attribute.
        """
        # Extract the human message from langchain's message format
        user_message = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                user_message = msg.content
                break
            elif hasattr(msg, 'content') and not hasattr(msg, 'type'):
                # Fallback if it's just a string or simple object
                user_message = msg.content if hasattr(msg, 'content') else str(msg)
                break

        if not user_message:
            user_message = str(messages[-1]) if messages else ""

        # Prepare the API endpoint and payload
        endpoint = f"{self.base_url}/api/v1/workspace/{self.workspace_slug}/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "message": user_message,
            "mode": "chat"  # 'chat' maintains history, 'query' does not
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            # Extract the text response from AnythingLLM's JSON
            answer = data.get('textResponse', '')
            # Create a simple response object compatible with the rest of the code
            return type('Response', (), {'content': answer})()
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling AnythingLLM API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nResponse: {e.response.text}"
            return type('Response', (), {'content': error_msg})()


# ------------------------------------------------------------------
# 2. Configuration and prompt (same as before)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert accounting AI. Extract all financial information from the given text and return a structured JSON object.

The JSON must have exactly this format:
{
  "trial_balance": [
    {"account_name": "Cash", "debit": 1000.0, "credit": 0.0},
    {"account_name": "Sales Revenue", "debit": 0.0, "credit": 5000.0}
  ]
}

Rules:
- Only include accounts that appear in the text (explicit numbers).
- Use standard accounting names: Assets, Liabilities, Equity, Revenue, Expenses.
- If an amount is an expense, put it in debit; if revenue, credit.
- Do not invent numbers. If a number is ambiguous, skip it.
- Return ONLY valid JSON, no extra text.
"""

USER_PROMPT_TEMPLATE = """Extract financial data from the following text:

{text}

Return the JSON as specified."""

# ------------------------------------------------------------------
# 3. File text extraction functions (unchanged)
# ------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = Document(tmp.name)
        return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_image(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        img = Image.open(tmp.name)
        text = pytesseract.image_to_string(img)
    return text

def extract_text(file_bytes: bytes, file_type: str) -> str:
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_bytes)
    elif file_type == "text/plain":
        return extract_text_from_txt(file_bytes)
    elif file_type.startswith("image/"):
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# ------------------------------------------------------------------
# 4. LLM extraction using AnythingLLM
# ------------------------------------------------------------------
def extract_financial_data(text: str, llm_client: ChatAnythingLLM) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text[:15000])}
    ]
    # Convert messages to LangChain-like format for our custom client
    class MockMessage:
        def __init__(self, content, type=None):
            self.content = content
            self.type = type

    langchain_messages = [
        MockMessage(SYSTEM_PROMPT, type="system"),
        MockMessage(USER_PROMPT_TEMPLATE.format(text=text[:15000]), type="human")
    ]

    response = llm_client.invoke(langchain_messages)
    content = response.content.strip()

    # Parse JSON from response
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        data = json.loads(content)
        return data.get("trial_balance", [])
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON from LLM response: {e}")
        st.text("Raw response: " + content[:500])
        return []

# ------------------------------------------------------------------
# 5. Spreadsheet builder (unchanged)
# ------------------------------------------------------------------
def build_spreadsheet(trial_balance: List[Dict[str, Any]]) -> BytesIO:
    # Create Trial Balance sheet
    df_tb = pd.DataFrame(trial_balance)
    # Ensure columns exist
    for col in ["account_name", "debit", "credit"]:
        if col not in df_tb.columns:
            df_tb[col] = 0.0
    df_tb = df_tb[["account_name", "debit", "credit"]]

    # Compute totals
    total_debit = df_tb["debit"].sum()
    total_credit = df_tb["credit"].sum()
    df_tb.loc["Total"] = ["", total_debit, total_credit]

    # Build P&L (Revenue and Expense accounts)
    revenue_accounts = df_tb[df_tb["account_name"].str.contains("revenue|sales|income", case=False, na=False)]
    expense_accounts = df_tb[df_tb["account_name"].str.contains("expense|cost|cogs", case=False, na=False)]
    pnl = pd.DataFrame({
        "Revenue": revenue_accounts["credit"].values if not revenue_accounts.empty else [0],
        "Expenses": expense_accounts["debit"].values if not expense_accounts.empty else [0]
    })
    net_income = pnl["Revenue"].sum() - pnl["Expenses"].sum()
    pnl.loc["Total"] = [pnl["Revenue"].sum(), pnl["Expenses"].sum()]
    pnl.loc["Net Income"] = [net_income, ""]

    # Build simple Balance Sheet (Assets = Liabilities + Equity)
    asset_accounts = df_tb[df_tb["account_name"].str.contains("asset|cash|inventory|receivable", case=False, na=False)]
    liability_accounts = df_tb[df_tb["account_name"].str.contains("liability|payable|debt|loan", case=False, na=False)]
    equity_accounts = df_tb[df_tb["account_name"].str.contains("equity|capital|retained", case=False, na=False)]

    bs = pd.DataFrame({
        "Assets": asset_accounts["debit"].values if not asset_accounts.empty else [0],
        "Liabilities": liability_accounts["credit"].values if not liability_accounts.empty else [0],
        "Equity": equity_accounts["credit"].values if not equity_accounts.empty else [0]
    })
    bs.loc["Total"] = [bs["Assets"].sum(), bs["Liabilities"].sum(), bs["Equity"].sum()]

    # Write to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_tb.to_excel(writer, sheet_name="Trial Balance", index=False)
        pnl.to_excel(writer, sheet_name="Profit & Loss", index=False)
        bs.to_excel(writer, sheet_name="Balance Sheet", index=False)
    output.seek(0)
    return output

# ------------------------------------------------------------------
# 6. Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Financial Extractor to Spreadsheet", layout="wide")
st.title("📊 AI Financial Extractor → Excel Accounts")
st.markdown("Upload any document (PDF, DOCX, TXT, image) and get a ready‑to‑use accounting spreadsheet.")

with st.sidebar:
    st.header("🔗 AnythingLLM Connection Settings")
    base_url = st.text_input("AnythingLLM Base URL", value="http://localhost:3001",
                             help="e.g., http://localhost:3001 or http://127.0.0.1:3001")
    api_key = st.text_input("API Key", type="password")
    workspace_slug = st.text_input("Workspace Slug", value="main_workspace",
                                   help="Find this in AnythingLLM settings")

    if st.button("Test Connection"):
        if not api_key:
            st.error("Please enter an API Key")
        elif not workspace_slug:
            st.error("Please enter a Workspace Slug")
        else:
            try:
                test_client = ChatAnythingLLM(base_url, api_key, workspace_slug)
                test_response = test_client.invoke([{"role": "user", "content": "Hello, respond with 'OK' if you receive this."}])
                if test_response.content and "error" not in test_response.content.lower():
                    st.success("✅ Connection successful!")
                else:
                    st.error(f"Connection failed: {test_response.content[:200]}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    st.divider()
    st.caption("🔒 **Privacy**: All processing happens on your local AnythingLLM server. No data leaves your computer.")

# Main area
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    help="PDF, Word, text, or scanned image (OCR)"
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type

    with st.spinner("📄 Extracting text from file..."):
        try:
            raw_text = extract_text(file_bytes, file_type)
            st.success("Text extraction complete.")
            with st.expander("Preview extracted text (first 1000 chars)"):
                st.text(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))
        except Exception as e:
            st.error(f"Text extraction failed: {e}")
            st.stop()

    if st.button("🧠 Extract Financial Information & Build Spreadsheet", type="primary"):
        if not api_key:
            st.warning("Please enter your API Key in the sidebar.")
        elif not workspace_slug:
            st.warning("Please enter your Workspace Slug in the sidebar.")
        else:
            with st.spinner("🤖 AI is reading financial data... (may take up to 30 seconds)"):
                try:
                    # Create AnythingLLM client
                    llm_client = ChatAnythingLLM(base_url, api_key, workspace_slug)

                    # Extract financial data
                    trial_balance = extract_financial_data(raw_text, llm_client)

                    if not trial_balance:
                        st.warning("No financial accounts detected. The document may not contain structured financial numbers.")
                    else:
                        # Show extracted accounts
                        st.subheader("📋 Extracted Trial Balance")
                        df_display = pd.DataFrame(trial_balance)
                        st.dataframe(df_display, use_container_width=True)

                        # Build Excel
                        excel_file = build_spreadsheet(trial_balance)
                        st.download_button(
                            label="📥 Download Excel Spreadsheet",
                            data=excel_file,
                            file_name="financial_statements.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"LLM extraction failed: {e}")
                    st.info("Check that your AnythingLLM server is running and the workspace slug is correct.")
