import pandas as pd
from deep_translator import GoogleTranslator
import time

# --- CẤU HÌNH ---
INPUT_FILE = "./data/voz_data_filtered.csv"
OUTPUT_FILE = "./data/voz_data_english.csv"

# 1. TỪ ĐIỂN MAP SLANG (Quan trọng nhất để giữ Sentiment)
# Model tiếng Anh sẽ không hiểu "reset" là "tự tử", nên ta phải map tay trước.
SLANG_MAPPING = {
    # Nhóm tiêu cực mạnh (Depression/Suicidal)
    "reset": "commit suicide",
    "đăng xuất": "die",
    "ra đi": "die",
    "nhảy cầu": "jump off a bridge",
    "nhảy lầu": "jump off a building",
    "kết thúc cuộc đời": "end my life",
    "không muốn sống": "do not want to live",
    "bế tắc": "hopeless",
    "trầm cảm": "depression",
    "áp lực": "pressure",
    "stress": "stress",
    
    # Nhóm học tập (Academic)
    "rớt môn": "fail the exam",
    "tạch môn": "fail the subject",
    "nợ môn": "fail the subject",
    "học lại": "retake the course",
    "bị đuổi": "expelled",
    "cảnh cáo học vụ": "academic warning",
    "ra trường": "graduate",
    "đồ án": "thesis",
    "bảo vệ": "defense",
    "điểm kém": "bad grades",
    "mất gốc": "lost basic knowledge",
    "ngu học": "stupid at studying",
    "deadline dí": "deadline chasing"
}

def map_vietnamese_slang(text):
    """Thay thế các từ lóng tiếng Việt bằng từ tiếng Anh tương đương"""
    if not isinstance(text, str): return ""
    
    # Chuyển về chữ thường để map cho dễ
    text_lower = text.lower()
    
    for vn_word, en_word in SLANG_MAPPING.items():
        if vn_word in text_lower:
            # Replace đơn giản (có thể cải tiến bằng regex nếu cần chính xác hơn)
            text_lower = text_lower.replace(vn_word, en_word)
            
    return text_lower

def translate_batch(text_series):
    """
    Hàm dịch cả cột dữ liệu.
    Dùng GoogleTranslator.
    """
    translator = GoogleTranslator(source='vi', target='en')
    results = []
    total = len(text_series)
    
    print(f"Bắt đầu dịch {total} dòng (Sẽ mất một chút thời gian)...")
    
    for i, text in enumerate(text_series):
        try:
            # 1. Xử lý slang trước
            precessed_text = map_vietnamese_slang(text)
            
            # 2. Cắt ngắn nếu quá dài (Google Translate giới hạn ~5000 ký tự)
            if len(precessed_text) > 4500:
                precessed_text = precessed_text[:4500]
            
            # 3. Dịch
            # Nếu text rỗng hoặc quá ngắn thì bỏ qua
            if len(precessed_text) < 3: 
                translated = ""
            else:
                translated = translator.translate(precessed_text)
            
            results.append(translated)
            
            # In tiến độ mỗi 10 dòng
            if (i + 1) % 10 == 0:
                print(f" -> Đã dịch: {i + 1}/{total} dòng")
                
        except Exception as e:
            print(f" -> Lỗi dòng {i}: {e}")
            results.append(None) # Đánh dấu lỗi để lọc sau
            
    return results

# --- MAIN EXECUTION ---
try:
    # 1. Load dữ liệu
    print("Đang đọc file dữ liệu...")
    df = pd.read_csv(INPUT_FILE)
    
    # Fill NaN bằng chuỗi rỗng để tránh lỗi
    df['full_text'] = df['full_text'].fillna('')
    
    # 2. Thực hiện dịch
    # Ta dịch cột 'full_text' (Tiêu đề + Nội dung) để có ngữ cảnh đầy đủ nhất
    df['translated_text'] = translate_batch(df['full_text'])
    
    # 3. Làm sạch sau khi dịch
    # Bỏ các dòng dịch lỗi (None) hoặc rỗng
    df_clean = df.dropna(subset=['translated_text'])
    df_clean = df_clean[df_clean['translated_text'] != ""]
    
    # 4. Lưu kết quả
    # Chỉ giữ lại các cột cần thiết cho Model
    final_cols = ['id', 'created_utc', 'date_readable', 'translated_text', 'full_text']
    df_clean[final_cols].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("HOÀN TẤT!")
    print(f"File kết quả: {OUTPUT_FILE}")
    print(f"Số lượng mẫu sẵn sàng cho Model: {len(df_clean)}")
    print("="*50)
    
    # Xem thử 3 dòng đầu
    print(df_clean[['full_text', 'translated_text']].head(3))

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'. Hãy chạy script crawl trước!")