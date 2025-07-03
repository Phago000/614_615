import streamlit as st
import os
import fitz  # PyMuPDF
import tempfile
import time
import re
import io
import zipfile
from PIL import Image
import google.generativeai as genai
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# --- CONFIGURATION ---

# Attempt to get the API key from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    GEMINI_API_KEY = ""  # Use an empty string if not found

# A blocklist of common 3-letter words to ignore to prevent false positives.
BLOCKLIST = {
    'THE', 'AND', 'ALL', 'FOR', 'CTS', 'USD', 'HKD', 'MFG', 
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
}

# --- SESSION STATE INITIALIZATION ---

if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = time.time()
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_file_for_preview' not in st.session_state:
    st.session_state.selected_file_for_preview = None

# --- HELPER AND CORE FUNCTIONS ---

def add_log(message, level="info"):
    """Adds a log message to the session state for debugging."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })

def check_rate_limit():
    """Checks the API call rate limit to avoid errors."""
    current_time = time.time()
    if current_time - st.session_state.last_reset_time >= 60:
        st.session_state.api_calls_count = 0
        st.session_state.last_reset_time = current_time
    
    if st.session_state.api_calls_count >= 60:
        raise Exception("API rate limit exceeded. Please wait a moment.")
    
    st.session_state.api_calls_count += 1

# --- MODIFIED CORE LOGIC ---

def extract_code_with_rules(text):
    """
    Extracts the agency code from page text using a prioritized set of rules.
    This is the primary, high-accuracy method.
    """
    if not text:
        return None

    # Rule 1: Find a 3-letter code at the top-left of the page (most reliable for Rpt_615).
    # It's usually one of the first words in the document.
    lines = text.split('\n')
    if lines:
        first_line_words = lines[0].strip().split()
        if first_line_words and len(first_line_words[0]) == 3 and first_line_words[0].isupper() and first_line_words[0] not in BLOCKLIST:
            code = first_line_words[0]
            add_log(f"Rule 1 (Top-Left Code): Found '{code}'")
            return code

    # Rule 2: Find the code in the "WA Code" column (reliable backup).
    # Pattern looks for a 3-letter code followed by numbers, as a whole word.
    wa_code_match = re.search(r'\b([A-Z]{3})\d{3,}\b', text)
    if wa_code_match:
        code = wa_code_match.group(1)
        if code not in BLOCKLIST:
            add_log(f"Rule 2 (WA Code Column): Found '{code}'")
            return code

    # Rule 3: Find the code in the report title (handles Rpt_614 and other formats).
    title_pattern = r'(?:RECEIVED|OUTSTANDING)\s+FEES\s+REPORT\s+([A-Z]{3})'
    title_match = re.search(title_pattern, text.upper().replace('\n', ' '))
    if title_match:
        code = title_match.group(1)
        if code not in BLOCKLIST:
            add_log(f"Rule 3 (Report Title): Found '{code}'")
            return code
            
    return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def make_api_call(model, prompt, image):
    """Makes a call to the Gemini API with automatic retries."""
    check_rate_limit()
    try:
        return model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(
            temperature=0
        ))
    except Exception as e:
        if "429" in str(e): # Specific handling for rate limit errors
            time.sleep(5)
        raise e

def extract_code_with_ai(pdf_path, page_number, api_key, status_text):
    """
    Uses enhanced rules and an adaptive AI model to identify the agency code.
    It prioritizes the rule-based method and uses AI as a fallback.
    """
    doc = None
    page_text = ""
    
    try:
        # Step 1: Extract text from the PDF page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        page_text = page.get_text()
        doc.close()

        # Step 2: Use the new, powerful rule-based extraction function first
        rule_based_code = extract_code_with_rules(page_text)

        # Step 3: Decision Logic
        # If the rules find a code, we trust it. It's fast, cheap, and highly reliable.
        if rule_based_code:
            add_log(f"Page {page_number+1}: Decision - Trusting reliable rule-based result '{rule_based_code}'.")
            return {"code": rule_based_code, "method": "rules_optimized", "confidence": "high", "text": page_text}

        # If rules fail AND we have an API key, use AI as a fallback.
        if api_key:
            status_text.text(f"Rules failed for page {page_number+1}. Using AI analysis...")
            page_image = convert_pdf_to_image(pdf_path, page_number)

            if not page_image:
                add_log(f"Page {page_number+1}: Could not convert to image for AI.", "warning")
                return {"code": "UNKNOWN", "method": "rules_failed", "confidence": "low", "text": page_text}

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- NEW ADAPTIVE AI PROMPT ---
            prompt = """You are a specialized financial document analyst. Your task is to find a single 3-letter agency code on the page.

            Follow these rules in order:
            1.  **Primary Location (Top-Left):** Look for a 3-letter code standing alone at the very top-left of the page (e.g., 'APO', 'FPL', 'OFS'). This is the most likely location.
            2.  **Secondary Location (WA Code Column):** If not found, look in the "WA Code" column. The code is the first 3 letters of entries like 'FPL007' or 'OFS030'.
            3.  **Tertiary Location (Title):** Look for the code in the main report title, which might be "Received Fees Report XXX" or "Outstanding Fees Report XXX".

            **CRITICAL RULES TO IGNORE:**
            -   **NEVER** use 'WHK'. It is part of an Account Number.
            -   **NEVER** use 'ALL', 'USD', 'HKD', or any currency/month code.
            -   **NEVER** use 'CTS' or any part of the footer/header like 'Print Date'.

            The code is always exactly 3 uppercase letters. Analyze the page and return ONLY a JSON object with your finding. If no valid code is found, return "UNK".

            Example Response:
            {
                "code": "FPL"
            }
            """
            
            response = make_api_call(model, prompt, page_image)
            
            try:
                # Robust JSON parsing from AI response
                json_str = response.text
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')
                
                ai_results = json.loads(json_str.strip())
                ai_code = ai_results.get('code', 'UNK').upper()
                add_log(f"Page {page_number+1}: AI analysis returned '{ai_code}'.")
                final_code = ai_code if ai_code not in ["UNK", ""] else "UNKNOWN"
                return {"code": final_code, "method": "ai_fallback", "confidence": "medium", "text": page_text}
            except Exception as e:
                add_log(f"Page {page_number+1}: AI response parsing failed: {e}", "error")
                return {"code": "UNKNOWN", "method": "ai_error", "confidence": "low", "text": page_text}

        # If rules fail and there's no API key, we mark as unknown.
        add_log(f"Page {page_number+1}: Rules failed and no API key available.", "warning")
        return {"code": "UNKNOWN", "method": "rules_failed_no_api", "confidence": "low", "text": page_text}

    except Exception as e:
        add_log(f"Critical error processing page {page_number+1}: {str(e)}", "error")
        return {"code": "UNKNOWN", "method": "error", "confidence": "low", "text": page_text}
    finally:
        if doc:
            doc.close()

# --- UNCHANGED UTILITY AND PROCESSING FUNCTIONS ---

def convert_pdf_to_image(pdf_path, page_num):
    """Converts a PDF page to a high-quality image for AI analysis."""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        zoom = 4  # Increased zoom for better clarity
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        add_log(f"PDF to image conversion error: {str(e)}", "error")
        return None
    finally:
        if doc:
            doc.close()

def is_summary_page(text):
    """Checks if a page is a summary or final page."""
    if not text:
        return False
    summary_indicators = ["End of Report", "End of Reoprt", "Summary", "Grand Total"]
    # Check if any indicator is present, case-insensitively
    return any(indicator.lower() in text.lower() for indicator in summary_indicators)

def determine_report_type(page_text):
    """Determines the report type to assist in filename generation."""
    if "Received Fees Report" in page_text:
        return {'report_num': '615', 'format': 'MF'}
    elif "Outstanding" in page_text:
        return {'report_num': '614', 'format': 'Outstanding'}
    return {'report_num': '615', 'format': 'MF'} # Default

def generate_filename(code, page_text):
    """Generates a standardized filename based on the code and report type."""
    report_info = determine_report_type(page_text)
    report_num = report_info['report_num']
    format_type = report_info['format']
    
    if report_num == '614':
        return f"Rpt 614-{code} {format_type}"
    else:
        return f"Rpt_{report_num}_{code}_MF"

def create_zip_buffer(generated_files):
    """Creates a ZIP file in memory containing all generated PDFs."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in generated_files:
            zip_file.writestr(file['filename'], file['content'])
    zip_buffer.seek(0)
    return zip_buffer

def process_pdf(uploaded_file, progress_bar, status_text):
    """Main function to orchestrate the PDF splitting process."""
    temp_path = None
    try:
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        
        if not GEMINI_API_KEY:
            st.warning("Gemini API key not configured. Accuracy may be reduced. Using rule-based extraction only.")
        
        st.session_state.generated_files = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        doc.close()
        
        page_codes = []
        
        # Step 1: Identify codes for all pages
        for page_num in range(total_pages):
            progress = (page_num + 1) / (total_pages * 2)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing page {page_num + 1}/{total_pages}...")
            
            doc = fitz.open(temp_path)
            page_text = doc[page_num].get_text()
            doc.close()
            
            if is_summary_page(page_text):
                page_codes.append({'page_num': page_num, 'code': 'SUMMARY', 'text': page_text})
                continue
            
            code_info = extract_code_with_ai(temp_path, page_num, GEMINI_API_KEY, status_text)
            page_codes.append({
                'page_num': page_num,
                'code': code_info.get('code', 'UNKNOWN'),
                'text': code_info.get('text', ''),
            })
            time.sleep(0.05)
        
        # Step 2: Group consecutive pages with the same code
        page_groups = []
        current_group = []
        for page_info in page_codes:
            if page_info['code'] == 'SUMMARY':
                if current_group:
                    page_groups.append(current_group)
                    current_group = []
                continue
            
            if not current_group or page_info['code'] == current_group['code']:
                current_group.append(page_info)
            else:
                page_groups.append(current_group)
                current_group = [page_info]
        
        if current_group:
            page_groups.append(current_group)
        
        # Step 3: Create a new PDF for each group
        status_text.text("Merging pages and creating files...")
        for group_index, group in enumerate(page_groups):
            progress = 0.5 + ((group_index + 1) / (len(page_groups) * 2))
            progress_bar.progress(progress)
            
            if not group: continue
            
            code = group['code']
            filename = f"{generate_filename(code, group['text'])}.pdf"
            
            output_doc = fitz.open()
            source_doc = fitz.open(temp_path)
            for page_info in group:
                output_doc.insert_pdf(source_doc, from_page=page_info['page_num'], to_page=page_info['page_num'])
            
            # Save to a memory buffer
            pdf_bytes = output_doc.tobytes(garbage=4, clean=True, deflate=True)
            output_doc.close()
            source_doc.close()
            
            st.session_state.generated_files.append({
                'filename': filename,
                'content': pdf_bytes,
                'pages': [p['page_num'] for p in group],
                'code': code,
                'page_count': len(group),
            })
            time.sleep(0.05)
            
        st.session_state.processing_complete = True
        if st.session_state.generated_files:
            st.session_state.zip_data = create_zip_buffer(st.session_state.generated_files)
        
        return st.session_state.generated_files
    
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {str(e)}")
        return []
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                add_log(f"Failed to remove temp file: {e}", "error")

def set_preview_file(file_index):
    """Sets the file to be displayed in the preview pane."""
    st.session_state.selected_file_for_preview = file_index

# --- STREAMLIT UI LAYOUT ---

st.set_page_config(layout="wide")
st.title("PDF Report Splitter")
st.write("This tool splits multi-page PDF reports into separate files based on an agency code.")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1.  **Upload** your PDF file.
    2.  Click the **Process PDF** button.
    3.  **Wait** for the processing to complete.
    4.  **Download** your split files individually or as a single ZIP file.
    """)
    st.markdown("---")
    st.subheader("Features")
    st.markdown("""
    - Automatically groups consecutive pages with the same code.
    - Creates a separate PDF for each agency code group.
    - Ignores summary and end-of-report pages.
    """)
    st.markdown("---")
    st.subheader("System Status")
    if GEMINI_API_KEY:
        st.success("✅ Gemini API is configured.")
    else:
        st.warning("⚠️ Gemini API not configured.")
        st.info("The tool will rely on rule-based extraction only. For best results, configure the API key.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.info(f"Uploaded: **{uploaded_file.name}** ({round(uploaded_file.size/1024, 1)} KB)")
    
    if st.button("Process PDF", key="process_button", use_container_width=True, type="primary"):
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        st.session_state.selected_file_for_preview = None
        
        progress_container = st.container()
        with progress_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("Starting process...")
            
            generated_files = process_pdf(uploaded_file, progress_bar, status_text)
            
            progress_bar.progress(1.0)
            if generated_files:
                status_text.success(f"Processing complete! {len(generated_files)} files were generated.")
            else:
                status_text.error("Processing finished, but no files were generated. Please check the PDF content.")

if st.session_state.processing_complete and st.session_state.generated_files:
    st.markdown("---")
    st.header(f"Processing Results ({len(st.session_state.generated_files)} files)")
    
    if st.session_state.zip_data:
        st.download_button(
            label=f"⬇️ Download All as ZIP ({len(st.session_state.generated_files)} files)",
            data=st.session_state.zip_data,
            file_name=f"{os.path.splitext(uploaded_file.name)}_split.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    col1, col2 = st.columns()

    with col1:
        st.subheader("Generated Files")
        # Group files by code for organized display
        grouped_files = {}
        for i, file in enumerate(st.session_state.generated_files):
            code = file.get('code', 'UNKNOWN')
            if code not in grouped_files:
                grouped_files[code] = []
            grouped_files[code].append((i, file))
        
        for code, files in grouped_files.items():
            with st.expander(f"Code: {code} ({len(files)} file(s))", expanded=True):
                for file_idx, file in files:
                    c1, c2 = st.columns()
                    with c1:
                        page_range_str = f" (Original Pgs: {min(file['pages'])+1}-{max(file['pages'])+1})" if len(file['pages']) > 1 else f" (Original Pg: {file['pages']+1})"
                        if st.button(f"📄 {file['filename']}", key=f"preview_{file_idx}", use_container_width=True):
                            set_preview_file(file_idx)
                    with c2:
                        st.download_button(
                            label="Download",
                            data=file['content'],
                            file_name=file['filename'],
                            mime="application/pdf",
                            key=f"download_{file_idx}"
                        )

    with col2:
        st.subheader("Preview")
        if st.session_state.selected_file_for_preview is not None:
            file_idx = st.session_state.selected_file_for_preview
            file = st.session_state.generated_files[file_idx]
            
            st.info(f"Showing preview for: **{file['filename']}**")
            
            try:
                doc = fitz.open(stream=file['content'], filetype="pdf")
                max_preview_pages = min(5, doc.page_count)
                
                st.write(f"Previewing first {max_preview_pages} of {doc.page_count} page(s):")
                
                for page_idx in range(max_preview_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(dpi=150) # Use DPI for better quality preview
                    st.image(pix.tobytes("png"), caption=f"Page {page_idx+1}")
                
                doc.close()
            except Exception as e:
                st.error(f"Could not display preview: {e}")
        else:
            st.info("Click on a file from the list on the left to see a preview here.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 PDF Report Splitter</div>", unsafe_allow_html=True)
