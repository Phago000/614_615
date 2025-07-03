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

# 嘗試從Streamlit secrets獲取API密鑰
try:
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    GEMINI_API_KEY = ""  # 如果無法從secrets獲取，則使用空字串

# 初始化會話狀態
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

# 常見機構代碼
COMMON_CODES = {'OFS', 'WMG', 'WCL', 'DOL', 'LNI', 'DFW', 'DOR', 'ECY', 'WSP', 'DOH', 'FPL', 'IPP'}

# 添加日誌消息函數（僅內部使用，不顯示在UI上）
def add_log(message, level="info"):
    """添加日誌消息到會話狀態"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })

def check_rate_limit():
    """檢查API速率限制"""
    current_time = time.time()
    if current_time - st.session_state.last_reset_time >= 60:  # 60秒重置
        st.session_state.api_calls_count = 0
        st.session_state.last_reset_time = current_time
    
    if st.session_state.api_calls_count >= 60:  # 每分鐘最多60次
        remaining_time = 60 - (current_time - st.session_state.last_reset_time)
        raise Exception(f"超過API速率限制，請稍後再試。")
    
    st.session_state.api_calls_count += 1

# 文本提取代碼函數 (Original Version)
def extract_code_from_text(text):
    """
    從文本中提取機構代碼 - 針對 Rpt_614 格式進行了優化。
    此版本會優先檢查報表標題。
    """
    if not text:
        return None
    
    # 標準化文本以便匹配
    upper_text = ' '.join(text.split()).upper()

    # 規則 1: 優先從報表標題中提取代碼 (最可靠的來源)
    # 匹配 "Outstanding Fees Report XXX" 或 "WCL/WMG Outstanding Fees Report"
    title_patterns = [
        r'(?:OUTSTANDING FEES REPORT|FEES REPORT)\s+([A-Z]{3})',
        r'([A-Z]{3})\s+OUTSTANDING FEES REPORT'
    ]
    for pattern in title_patterns:
        match = re.search(pattern, upper_text)
        if match:
            code = match.group(1)
            if code not in ['THE', 'AND', 'ALL', 'FOR', 'CTS']:
                add_log(f"文本規則：從標題找到代碼 '{code}'")
                return code

    # 規則 2: 從 WA Code 數據列中提取 (第二可靠的來源)
    # 匹配換行符後的 "FPL007", "IPP021" 等格式
    wa_code_match = re.search(r'\n([A-Z]{3})\d+\s', text)
    if wa_code_match:
        code = wa_code_match.group(1)
        add_log(f"文本規則：從WA Code列找到代碼 '{code}'")
        return code

    # 規則 3: 檢查高優先度代碼是否單獨存在於頁面頂部
    if 'OFS' in upper_text[:300]: # 只搜索文件開頭部分
        return 'OFS'
    if 'WMG' in upper_text[:300]:
        return 'WMG'
    if 'WCL' in upper_text[:300]:
        return 'WCL'

    # 如果以上規則都失敗，返回 None
    return None

# AI調用函數 (UNCHANGED)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def make_api_call(model, prompt, image):
    """調用Gemini API（帶重試）"""
    check_rate_limit()
    try:
        return model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(
            temperature=0
        ))
    except Exception as e:
        if "429" in str(e):
            time.sleep(5)
        raise e

# 核心識別函數 (Original structure with MODIFIED PROMPT)
def extract_code_with_ai(pdf_path, page_number, api_key, status_text):
    """
    使用增強的AI和決策邏輯識別代碼 - 針對 Rpt_614 格式進行了特別優化。
    """
    doc = None
    page_text = ""
    text_extracted_code = None
    
    try:
        # 步驟 1: 使用優化後的文本規則提取
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        page_text = page.get_text()
        doc.close()
        text_extracted_code = extract_code_from_text(page_text)

        if text_extracted_code and not api_key:
            add_log(f"第 {page_number+1} 頁: 無API密鑰，使用文本規則找到 '{text_extracted_code}'。")
            return {"code": text_extracted_code, "method": "text_rule_only", "confidence": "high", "text": page_text}
        
        if not api_key:
            return {"code": "UNK", "method": "text_rule_only", "confidence": "low", "text": page_text}

        # 步驟 2: 準備並調用AI模型
        status_text.text(f"正在使用AI增強分析第 {page_number+1} 頁...")
        page_image = convert_pdf_to_image(pdf_path, page_number)

        if not page_image:
            code = text_extracted_code if text_extracted_code else "UNK"
            return {"code": code, "method": "text_rule_fallback", "confidence": "medium" if code != "UNK" else "low", "text": page_text}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # --- THIS IS THE ONLY CHANGE: THE NEW, ADAPTIVE PROMPT ---
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
        
        # 解析AI響應
        ai_code = "UNK"
        try:
            json_str = response.text
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')
            elif '```' in json_str:
                json_str = json_str.split('```').split('```')[0]
            
            ai_results = json.loads(json_str.strip())
            ai_code = ai_results.get('code', 'UNK').upper()
        except Exception as e:
            add_log(f"第 {page_number+1} 頁: AI響應解析失敗: {e}", "error")
            ai_code = "UNK"

        # 步驟 3: 最終決策 (Original Logic)
        if text_extracted_code:
            add_log(f"第 {page_number+1} 頁: 決策 - 信任優化後的文本規則結果 '{text_extracted_code}'。")
            final_code = text_extracted_code
            method = "text_rule_optimized"
            confidence = "high"
        elif ai_code != "UNK":
            add_log(f"第 {page_number+1} 頁: 決策 - 文本規則失敗，使用AI結果 '{ai_code}'。")
            final_code = ai_code
            method = "ai_fallback"
            confidence = "medium"
        else:
            add_log(f"第 {page_number+1} 頁: 決策 - 所有方法均失敗。")
            final_code = "UNK"
            method = "combined_failure"
            confidence = "low"
            
        return {"code": final_code, "method": method, "confidence": confidence, "text": page_text}

    except Exception as e:
        add_log(f"第 {page_number+1} 頁處理時發生嚴重錯誤: {str(e)}", "error")
        if text_extracted_code:
            return {"code": text_extracted_code, "method": "error_fallback_text", "confidence": "low", "text": page_text}
        return {"code": "UNK", "method": "error", "confidence": "low", "text": page_text}

def convert_pdf_to_image(pdf_path, page_num):
    """將PDF頁面轉換為圖像 - 增強清晰度版本"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        zoom = 6
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        add_log(f"PDF轉圖像錯誤: {str(e)}", "error")
        return None
    finally:
        if doc:
            doc.close()

def is_summary_page(text):
    """檢查頁面是否為摘要頁"""
    if not text:
        return False
    summary_indicators = ["End of Report", "End of Reoprt", "Summary", "Grand Total"]
    return any(indicator in text for indicator in summary_indicators)

def determine_report_type(page_text):
    """根據內容確定報表類型"""
    if "Received Fees Report" in page_text:
        return {'report_num': '615', 'format': 'MF'}
    elif "Outstanding" in page_text:
        return {'report_num': '614', 'format': 'Outstanding'}
    return {'report_num': '615', 'format': 'MF'}

def generate_filename(code, page_text):
    """生成適當的文件名"""
    report_info = determine_report_type(page_text)
    report_num = report_info['report_num']
    format_type = report_info['format']
    
    if report_num == '614':
        return f"Rpt 614-{code} {format_type}"
    else:
        return f"Rpt_{report_num}_{code}_MF"

def create_zip_buffer(generated_files):
    """創建包含所有文件的ZIP壓縮包"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in generated_files:
            zip_file.writestr(file['filename'], file['content'])
    zip_buffer.seek(0)
    return zip_buffer

def process_pdf(uploaded_file, progress_bar, status_text):
    """處理PDF文件"""
    temp_path = None
    
    try:
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        
        if not GEMINI_API_KEY:
            st.warning("未配置Gemini API密鑰，將僅使用文本規則識別代碼。請聯繫管理員設置API密鑰以提高識別準確率。")
        
        st.session_state.generated_files = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        doc.close()
        
        page_codes = []
        
        for page_num in range(total_pages):
            progress = (page_num + 1) / (total_pages * 2)
            progress_bar.progress(progress)
            status_text.text(f"識別第 {page_num + 1}/{total_pages} 頁的代碼...")
            
            doc = fitz.open(temp_path)
            page_text = doc[page_num].get_text()
            doc.close()
            
            if is_summary_page(page_text) or page_num == total_pages - 1:
                page_codes.append({'page_num': page_num, 'code': 'SUMMARY', 'text': page_text})
                continue
            
            code_info = extract_code_with_ai(temp_path, page_num, GEMINI_API_KEY, status_text)
            
            code = code_info.get('code', 'UNK')
            if code in ['UNK', 'ALL']:
                code = 'UNKNOWN'
            
            page_codes.append({
                'page_num': page_num,
                'code': code,
                'text': code_info.get('text', ''),
                'method': code_info.get('method', 'text_rule'),
                'confidence': code_info.get('confidence', 'medium')
            })
            time.sleep(0.1)
        
        page_groups = []
        current_group = []
        
        # Corrected Grouping Logic
        for i, page_info in enumerate(page_codes):
            if page_info['code'] == 'SUMMARY':
                if current_group:
                    page_groups.append(current_group)
                current_group = []
                continue
            
            # Compare with the code of the first item in the current group
            if not current_group or page_info['code'] == current_group[0]['code']:
                current_group.append(page_info)
            else:
                page_groups.append(current_group)
                current_group = [page_info]
        
        if current_group:
            page_groups.append(current_group)
        
        status_text.text("正在合併連續相同代碼的頁面...")
        
        for group_index, group in enumerate(page_groups):
            progress = 0.5 + ((group_index + 1) / (len(page_groups) * 2))
            progress_bar.progress(progress)
            
            if not group: continue
            
            code = group[0]['code']
            first_page_text = group[0]['text']
            
            base_filename = generate_filename(code, first_page_text)
            filename = f"{base_filename}.pdf"
            
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_output.close()
            
            source_doc = fitz.open(temp_path)
            output_doc = fitz.open()
            
            for page_info in group:
                page_num = page_info['page_num']
                output_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
            
            output_doc.save(temp_output.name, garbage=4, clean=True, deflate=True)
            output_doc.close()
            source_doc.close()
            
            with open(temp_output.name, 'rb') as file:
                file_content = file.read()
            
            st.session_state.generated_files.append({
                'filename': filename,
                'content': file_content,
                'pages': [p['page_num'] for p in group],
                'code': code,
                'page_count': len(group),
                'method': group[0].get('method', 'text_rule'),
                'confidence': group[0].get('confidence', 'medium')
            })
            
            try:
                os.remove(temp_output.name)
            except: pass
            time.sleep(0.1)
        
        st.session_state.processing_complete = True
        
        if st.session_state.generated_files:
            st.session_state.zip_data = create_zip_buffer(st.session_state.generated_files)
        
        return st.session_state.generated_files
    
    except Exception as e:
        st.error(f"處理PDF時出錯: {str(e)}")
        # For debugging, show the full error
        import traceback
        st.code(traceback.format_exc())
        return []
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except: pass

def set_preview_file(file_index):
    """設置要預覽的文件"""
    st.session_state.selected_file_for_preview = file_index

# --- Streamlit UI (Unchanged) ---
st.title("PDF報表拆分工具")
st.write("此工具可以將多頁PDF報表按機構代碼拆分成單獨的文件")

with st.sidebar:
    st.header("使用說明")
    st.markdown("1. 上傳PDF文件")
    st.markdown("2. 點擊「處理PDF」按鈕")
    st.markdown("3. 等待處理完成")
    st.markdown("4. 下載拆分後的文件")
    st.markdown("---")
    st.subheader("功能說明")
    st.markdown("- 連續的相同代碼頁面會自動合併為一個PDF文件")
    st.markdown("- 不同代碼的頁面會分開成獨立的PDF文件")
    st.markdown("- 摘要頁面會被自動忽略")
    st.markdown("---")
    st.subheader("兼容性說明")
    st.info("如果您在Adobe查看器中遇到顯示問題，請嘗試使用Chrome或Edge瀏覽器打開生成的文件")
    st.markdown("---")
    st.subheader("系統狀態")
    if GEMINI_API_KEY:
        st.success("✅ Gemini API已配置")
    else:
        st.warning("⚠️ Gemini API未配置")
        st.info("請聯繫管理員配置API以提高識別準確率")

uploaded_file = st.file_uploader("選擇PDF文件", type="pdf")

if uploaded_file is not None:
    st.info(f"已上傳: {uploaded_file.name} ({round(uploaded_file.size/1024, 1)} KB)")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        doc = fitz.open(temp_path)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
        img_data = pix.tobytes("png")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("文件首頁預覽:")
        with col2:
            st.image(img_data)
        
        doc.close()
        os.remove(temp_path)
    except Exception as e:
        st.warning(f"無法顯示預覽")

    process_button = st.button("處理PDF", key="process_button", use_container_width=True)
    
    if process_button:
        st.session_state.processing_complete = False
        st.session_state.zip_data = None
        st.session_state.selected_file_for_preview = None
        
        progress_container = st.container()
        with progress_container:
            st.write("處理進度:")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("準備處理...")
            
            generated_files = process_pdf(uploaded_file, progress_bar, status_text)
            
            progress_bar.progress(1.0)
            status_text.text(f"處理完成! 已生成 {len(generated_files)} 個文件。")
            
            if len(generated_files) > 0:
                st.success("✓ 處理成功! 請點擊下方的下載按鈕獲取處理後的文件。")
                st.info("💡 提示: 如果在Adobe中查看文件有問題，請嘗試使用Chrome或Edge打開。")

if st.session_state.processing_complete and st.session_state.generated_files:
    st.markdown("---")
    st.subheader(f"處理結果 (共 {len(st.session_state.generated_files)} 個文件)")
    
    if st.session_state.zip_data:
        st.download_button(
            label=f"下載所有文件 (ZIP包含 {len(st.session_state.generated_files)} 個文件)",
            data=st.session_state.zip_data,
            file_name="processed_files.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    grouped_files = {}
    for i, file in enumerate(st.session_state.generated_files):
        code = file.get('code', 'UNK')
        if code not in grouped_files:
            grouped_files[code] = []
        grouped_files[code].append((i, file))
    
    tabs = st.tabs([f"{code} ({len(files)})" for code, files in grouped_files.items()])
    
    for i, (code, files) in enumerate(grouped_files.items()):
        with tabs[i]:
            st.write("點擊檔案名稱來預覽文件:")
            
            for file_idx, file in files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    page_range = f"第{min(file['pages'])+1}-{max(file['pages'])+1}頁" if len(file['pages']) > 1 else f"第{file['pages'][0]+1}頁"
                    if st.button(f"{file['filename']} ({page_range}, 共{file['page_count']}頁)", key=f"file_{file_idx}", use_container_width=True):
                        set_preview_file(file_idx)
                with col2:
                    st.download_button(
                        label="下載",
                        data=file['content'],
                        file_name=file['filename'],
                        mime="application/pdf",
                        key=f"download_{file_idx}"
                    )
            
            st.info(f"機構代碼 {code}: {len(files)} 個文件")
    
    if st.session_state.selected_file_for_preview is not None:
        file_idx = st.session_state.selected_file_for_preview
        if 0 <= file_idx < len(st.session_state.generated_files):
            file = st.session_state.generated_files[file_idx]
            
            st.markdown("---")
            st.subheader(f"預覽: {file['filename']}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file['content'])
                temp_path = tmp_file.name
            
            try:
                doc = fitz.open(temp_path)
                max_preview_pages = min(5, doc.page_count)
                
                st.write(f"預覽前 {max_preview_pages} 頁 (共 {doc.page_count} 頁):")
                
                for page_idx in range(max_preview_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
                    img_data = pix.tobytes("png")
                    st.image(img_data, caption=f"第 {page_idx+1} 頁")
                
                doc.close()
            except Exception as e:
                st.error(f"顯示預覽時出錯")
            finally:
                try:
                    os.remove(temp_path)
                except: pass

st.markdown("---")
st.markdown("© 2025 PDF報表拆分工具")
