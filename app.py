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
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    GEMINI_API_KEY = ""

# Blocklist of common 3-letter words to prevent false positives
BLOCKLIST = {
    'THE', 'AND', 'ALL', 'FOR', 'CTS', 'USD', 'HKD', 'MFG', 
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'PRI'
}

# --- SESSION STATE INITIALIZATION ---
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_file_for_preview' not in st.session_state:
    st.session_state.selected_file_for_preview = None
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None


# --- HELPER FUNCTIONS ---
def add_log(message, level="info"):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append({"timestamp": timestamp, "message": message, "level": level})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def make_api_call(model, prompt, image):
    return model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(temperature=0))

# --- CORE IDENTIFICATION LOGIC (REBUILT FOR ACCURACY) ---

def extract_code_with_rules(text):
    """
    Extracts the agency code using a prioritized set of robust rules.
    THIS IS THE NEW, SMARTER TEXT-BASED FUNCTION.
    """
    if not text:
        return None

    # Rule 1: Find a 3-letter code at the very top-left of the page.
    # This is the most reliable pattern for Rpt_615.
    first_line = text.split('\n', 1)[0].strip()
    if len(first_line) == 3 and first_line.isupper() and first_line not in BLOCKLIST:
        add_log(f"Rule 1 (Top-Left Code) found: '{first_line}'")
        return first_line

    # Rule 2: Find the code in the "WA Code" column (reliable backup).
    wa_code_match = re.search(r'\b([A-Z]{3})\d{3,}\b', text)
    if wa_code_match:
        code = wa_code_match.group(1)
        if code not in BLOCKLIST:
            add_log(f"Rule 2 (WA Code Column) found: '{code}'")
            return code
            
    # Rule 3: Find code in a report title (for other report formats).
    title_pattern = r'(?:RECEIVED|OUTSTANDING)\s+FEES\s+REPORT\s+([A-Z]{3})'
    title_match = re.search(title_pattern, text.upper().replace('\n', ' '))
    if title_match:
        code = title_match.group(1)
        if code not in BLOCKLIST:
            add_log(f"Rule 3 (Report Title) found: '{code}'")
            return code
            
    return None

def extract_code_with_ai(pdf_path, page_number, api_key, status_text):
    """
    Main identification function. It now heavily relies on the superior rule-based
    method and only uses AI as a last resort.
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        page_text = page.get_text()

        # Step 1: Use the new, powerful rule-based extraction first.
        rule_based_code = extract_code_with_rules(page_text)

        # Step 2: Decision - If the rules found a code, we are DONE. Trust it.
        if rule_based_code:
            add_log(f"Page {page_number+1}: Trusting high-confidence rule-based result '{rule_based_code}'.")
            return {"code": rule_based_code, "method": "rules_optimized", "confidence": "high", "text": page_text}

        # Step 3: If rules failed AND we have an API key, use AI as a fallback.
        if api_key:
            status_text.text(f"Rules failed for page {page_number+1}. Using AI analysis as a fallback...")
            page_image = convert_pdf_to_image(pdf_path, page_number)

            if not page_image:
                return {"code": "UNKNOWN", "method": "image_conversion_failed", "confidence": "low", "text": page_text}

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = """You are a specialized financial document analyst. Your task is to find a single 3-letter agency code on the page.

            Follow these rules in order:
            1.  **Primary Location (Top-Left):** Look for a 3-letter code standing alone at the very top-left of the page (e.g., 'APO', 'FPL', 'OFS'). This is the most likely location.
            2.  **Secondary Location (WA Code Column):** If not found, look in the "WA Code" column. The code is the first 3 letters of entries like 'FPL007' or 'OFS030'.

            **CRITICAL RULES TO IGNORE:**
            -   **NEVER** use 'WHK'. It is part of an Account Number.
            -   **NEVER** use 'ALL', 'USD', 'HKD', or any currency/month code.
            -   **NEVER** use 'CTS' or any part of the footer/header like 'Print Date'. The code is never 'PRI'.

            The code is always exactly 3 uppercase letters. Analyze the page and return ONLY a JSON object with your finding. If no valid code is found, return "UNK".
            Example: {"code": "FPL"}
            """
            
            response = make_api_call(model, prompt, page_image)
            
            try:
                json_str = re.search(r'\{.*\}', response.text, re.S).group(0)
                ai_results = json.loads(json_str)
                ai_code = ai_results.get('code', 'UNK').upper()
                final_code = ai_code if ai_code not in ["UNK", ""] else "UNKNOWN"
                return {"code": final_code, "method": "ai_fallback", "confidence": "medium", "text": page_text}
            except Exception as e:
                return {"code": "UNKNOWN", "method": "ai_error", "confidence": "low", "text": page_text}

        # If rules fail and there's no API key, mark as unknown.
        return {"code": "UNKNOWN", "method": "rules_failed_no_api", "confidence": "low", "text": page_text}

    finally:
        if doc:
            doc.close()

# --- UTILITY AND PROCESSING FUNCTIONS (UNCHANGED BUT INCLUDED FOR COMPLETENESS) ---

def convert_pdf_to_image(pdf_path, page_num):
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    finally:
        if doc:
            doc.close()

def is_summary_page(text):
    if not text: return False
    return any(indicator.lower() in text.lower() for indicator in ["End of Report", "End of Reoprt", "Summary", "Grand Total"])

def generate_filename(code, page_text):
    if "Received Fees Report" in page_text:
        return f"Rpt_615_{code}_MF"
    elif "Outstanding" in page_text:
        return f"Rpt_614_{code}_Outstanding"
    return f"Rpt_{code}_Misc"

def create_zip_buffer(generated_files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in generated_files:
            zip_file.writestr(file['filename'], file['content'])
    zip_buffer.seek(0)
    return zip_buffer

def process_pdf(uploaded_file, progress_bar, status_text):
    temp_path = None
    source_doc = None
    try:
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        st.session_state.generated_files = []

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        source_doc = fitz.open(temp_path)
        total_pages = len(source_doc)
        
        page_codes = []
        
        for page_num in range(total_pages):
            progress = (page_num + 1) / (total_pages * 2)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing page {page_num + 1}/{total_pages}...")
            
            page_text = source_doc[page_num].get_text()
            
            if is_summary_page(page_text):
                page_codes.append({'page_num': page_num, 'code': 'SUMMARY', 'text': page_text})
                continue
            
            code_info = extract_code_with_ai(temp_path, page_num, GEMINI_API_KEY, status_text)
            page_codes.append({
                'page_num': page_num,
                'code': code_info.get('code', 'UNKNOWN'),
                'text': code_info.get('text', ''),
            })
        
        page_groups = []
        current_group = []
        for page_info in page_codes:
            if page_info['code'] == 'SUMMARY':
                if current_group:
                    page_groups.append(current_group)
                current_group = []
                continue
            
            if not current_group or page_info['code'] == current_group[0]['code']:
                current_group.append(page_info)
            else:
                page_groups.append(current_group)
                current_group = [page_info]
        
        if current_group:
            page_groups.append(current_group)
        
        status_text.text("Merging pages and creating files...")
        for group_index, group in enumerate(page_groups):
            progress = 0.5 + ((group_index + 1) / (len(page_groups) * 2))
            progress_bar.progress(progress)
            
            if not group: continue
            
            code = group[0]['code']
            first_page_text = group[0]['text']
            filename = f"{generate_filename(code, first_page_text)}.pdf"
            
            output_doc = fitz.open()
            for page_info in group:
                output_doc.insert_pdf(source_doc, from_page=page_info['page_num'], to_page=page_info['page_num'])
            
            pdf_bytes = output_doc.tobytes(garbage=4, clean=True, deflate=True)
            output_doc.close()
            
            st.session_state.generated_files.append({
                'filename': filename,
                'content': pdf_bytes,
                'pages': [p['page_num'] for p in group],
                'code': code,
                'page_count': len(group),
            })
            
        st.session_state.processing_complete = True
        if st.session_state.generated_files:
            st.session_state.zip_data = create_zip_buffer(st.session_state.generated_files)
        
        return st.session_state.generated_files
    
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []
    
    finally:
        if source_doc:
            source_doc.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                add_log(f"Failed to remove temp file: {e}", "error")

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("PDF Report Splitter")
st.write("This tool splits multi-page PDF reports into separate files based on an agency code.")

with st.sidebar:
    st.header("Instructions")
    st.markdown("1. **Upload** your PDF file.\n2. Click the **Process PDF** button.\n3. **Wait** for processing.\n4. **Download** your split files.")
    st.divider()
    st.subheader("System Status")
    if GEMINI_API_KEY:
        st.success("✅ Gemini API is configured.")
    else:
        st.warning("⚠️ Gemini API not configured.")
        st.info("The tool will rely on rule-based extraction only.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.info(f"Uploaded: **{uploaded_file.name}**")
    
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
                status_text.error("Processing finished, but no files were generated. Please check the PDF content or errors above.")

if st.session_state.processing_complete and st.session_state.generated_files:
    st.divider()
    st.header(f"Processing Results ({len(st.session_state.generated_files)} files)")
    
    if st.session_state.zip_data:
        st.download_button(
            label=f"⬇️ Download All as ZIP ({len(st.session_state.generated_files)} files)",
            data=st.session_state.zip_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_split.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # UI for displaying files and previews
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generated Files")
        grouped_files = {}
        for i, file in enumerate(st.session_state.generated_files):
            code = file.get('code', 'UNKNOWN')
            if code not in grouped_files: grouped_files[code] = []
            grouped_files[code].append((i, file))
        
        for code, files in grouped_files.items():
            with st.expander(f"Code: {code} ({len(files)} file(s))", expanded=True):
                for file_idx, file in files:
                    c1, c2 = st.columns([3,1])
                    page_range = f"Pgs {min(file['pages'])+1}-{max(file['pages'])+1}" if len(file['pages']) > 1 else f"Pg {file['pages'][0]+1}"
                    c1.button(f"📄 {file['filename']} ({page_range})", key=f"preview_{file_idx}", on_click=set_preview_file, args=(file_idx,), use_container_width=True)
                    with c2:
                        st.download_button("Save", file['content'], file['filename'], "application/pdf", key=f"dl_{file_idx}", use_container_width=True)

    with col2:
        st.subheader("Preview")
        if st.session_state.selected_file_for_preview is not None:
            file_idx = st.session_state.selected_file_for_preview
            file = st.session_state.generated_files[file_idx]
            st.info(f"Showing preview for: **{file['filename']}**")
            try:
                doc = fitz.open(stream=file['content'], filetype="pdf")
                for i, page in enumerate(doc):
                    if i >= 3: # Limit preview to first 3 pages
                        st.write(f"...and {doc.page_count - i} more pages.")
                        break
                    pix = page.get_pixmap(dpi=150)
                    st.image(pix.tobytes("png"), caption=f"Page {i+1} of {doc.page_count}")
                doc.close()
            except Exception as e:
                st.error(f"Could not display preview: {e}")
        else:
            st.info("Click on a file from the list to see a preview here.")
