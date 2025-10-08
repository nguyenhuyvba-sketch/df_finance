import streamlit as st
import pandas as pd
from google import genai
from google.genai import types # Thêm import này để sử dụng GenerateContentConfig
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Khởi tạo State và Lịch sử Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dùng để theo dõi dữ liệu hiện tại và reset chat khi file mới được tải
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (cho Phân tích Tự động) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        # Lỗi API 400 INVALID_ARGUMENT (API key not valid) sẽ được bắt ở đây
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- CHỨC NĂNG BỔ SUNG: KHUNG CHAT HỎI ĐÁP ---
def chat_section(df_processed, api_key):
    """Xử lý giao diện và logic khung chat hỏi đáp với ngữ cảnh dữ liệu."""
    if not api_key:
        return # Không hiển thị khung chat nếu không có API key
    
    if df_processed is None:
        return # Không hiển thị khung chat nếu chưa có dữ liệu

    st.divider()
    st.subheader("6. HỎI ĐÁP VỚI CHUYÊN GIA AI (Chat) 💬")
    st.markdown("Bạn có thể đặt câu hỏi chi tiết về dữ liệu tài chính vừa được phân tích ở trên.")
    
    # Chuẩn bị Context Dữ liệu cho AI
    data_context = df_processed.to_markdown(index=False)
    
    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'
    
    # SYSTEM INSTRUCTION: Rất quan trọng để giữ ngữ cảnh và vai trò
    system_instruction = f"""
    Bạn là một chuyên gia phân tích tài chính AI thân thiện.
    Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng **CHỈ** dựa trên dữ liệu Báo cáo Tài chính đã được xử lý sau đây.
    Hãy phân tích cẩn thận các chỉ số tăng trưởng, tỷ trọng và các giá trị tuyệt đối.

    DỮ LIỆU TÀI CHÍNH ĐÃ XỬ LÝ (định dạng Markdown):
    {data_context}
    """
    
    # 1. Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Xử lý input mới
    if prompt := st.chat_input("Hỏi AI về dữ liệu tài chính của bạn..."):
        # Thêm prompt của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chuẩn bị nội dung cho API (chuyển đổi role Streamlit -> Gemini)
        gemini_contents = []
        for msg in st.session_state.messages:
            # Streamlit dùng 'assistant', Gemini dùng 'model'
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_contents.append({"role": role, "parts": [{"text": msg['content']}]})

        with st.chat_message("assistant"):
            with st.spinner("AI đang phân tích và trả lời..."):
                try:
                    # Cấu hình để truyền system_instruction 
                    chat_config = types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=gemini_contents,
                        config=chat_config # Tham số đúng để truyền system_instruction
                    )
                    ai_response = response.text
                except APIError as e:
                    ai_response = f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
                except Exception as e:
                    ai_response = f"Đã xảy ra lỗi không xác định trong quá trình chat: {e}"
                
                st.markdown(ai_response)
                # Thêm phản hồi của AI vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": ai_response})


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        # Kiểm tra và Reset Chat History nếu dữ liệu mới được tải
        if st.session_state.df_processed is None or not st.session_state.df_processed.equals(df_processed):
             st.session_state.messages = []
             st.session_state.df_processed = df_processed
             st.toast("Dữ liệu mới được tải. Lịch sử chat đã được làm mới.")

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, kiểm tra chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"
                    )
                with col2:
                    delta_value = (thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1) if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A",
                        delta=f"{delta_value:.2f}" if delta_value is not None else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except Exception as e:
                st.error(f"Lỗi khi tính toán chỉ số: {e}")
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích Tự động"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
            
            # --- Chức năng 6: Khung Chat Hỏi đáp ---
            api_key_chat = st.secrets.get("GEMINI_API_KEY")
            chat_section(df_processed, api_key_chat)


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
