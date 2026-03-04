import requests
import pandas as pd
import time
import random

# --- CẤU HÌNH BỘ LỌC (QUAN TRỌNG NHẤT) ---

# 1. Từ khóa để GỌI API (Tìm kiếm rộng)
SEARCH_KEYWORDS = [
    "thi", "đồ án", "nợ môn", "học lại", "ra trường", 
    "điểm kém", "gpa", "deadline", "áp lực học",
    "bỏ học", "rớt môn", "tạch môn"
]

# 2. Từ khóa để LỌC NỘI DUNG (Phải có ít nhất 1 từ trong nhóm này mới lấy)
# Dùng từ ghép để tránh nhầm lẫn (vd: "thi" -> "thi cuối kỳ")
ACADEMIC_KEYWORDS = [
    # Nhóm Thi cử & Kết quả
    "thi cuối kỳ", "thi giữa kỳ", "thi lại", "trả nợ môn", "học cải thiện",
    "thi thpt", "thi đại học", "xét tuyển", "nguyện vọng", "học bạ",
    "điểm thi", "điểm chuẩn", "đề thi", "ôn thi", "luyện thi",
    "khối a", "khối b", "khối c", "khối d", "đánh giá năng lực", "đgnl",
    
    # Nhóm Áp lực & Tốt nghiệp
    "đồ án tốt nghiệp", "khóa luận", "deadline dí",
    "rớt môn", "tạch môn", "nợ môn", "bảng điểm", "gpa", "cpa",
    "cảnh cáo học vụ", "bị đuổi học", "bảo lưu", "đăng ký tín chỉ",
    "ra trường", "thất nghiệp", "nộp bài", "phúc khảo", "qua môn",
    
    # Nhóm Tâm lý & Sentiment (Quan trọng cho Analysis)
    "áp lực đồng trang lứa", "peer pressure", "stress vì học",
    "reset", "đăng xuất", "tự tử", "nhảy lầu", "bế tắc",
    "mất gốc", "học kém", "ngu học", "chán học", "áp lực điểm số"
    
    # Đã loại bỏ: "sinh viên", "đại học", "giảng viên", "thầy cô" (quá chung chung)
]

# 3. Từ khóa LOẠI TRỪ (Gặp là xóa ngay - Chống Spam/QC)
BLACKLIST_KEYWORDS = [
    "tuyển dụng", "việc làm", "shopee", "lazada", "tiki", "thanh lý",
    "pass lại", "voucher", "nhượng phòng", "tìm trọ", "thi công", 
    "thiết kế", "xây dựng", "thi hoa hậu", "sex", "kèo bóng",
    "bằng lái", "sát hạch", "b2", "a1", "lái xe", "gplx", # Loại bỏ thi bằng lái
    "vay tiền", "tài chính", "chứng khoán", "bitcoin", "coin", "sale"
]

def strict_filter(text):
    """
    Hàm lọc cứng:
    1. Text phải đủ dài (> 30 ký tự).
    2. Không chứa từ khóa Blacklist.
    3. Phải chứa ít nhất 1 từ khóa Academic chuyên sâu.
    """
    if not isinstance(text, str): return False
    text_lower = text.lower()
    
    # Điều kiện 1: Độ dài
    if len(text_lower) < 30: return False
    
    # Điều kiện 2: Không chứa rác
    for bad_word in BLACKLIST_KEYWORDS:
        if bad_word in text_lower: return False
        
    # Điều kiện 3: Phải liên quan đến học tập
    is_academic = False
    for kw in ACADEMIC_KEYWORDS:
        if kw in text_lower:
            is_academic = True
            break
            
    return is_academic

def crawl_reddit_strict(subreddits=["vozforums", "TroChuyenLinhTinh", "VietNam"]):
    all_posts = {} # Dùng dict để tránh trùng lặp bài viết (theo ID)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    print(f"Bắt đầu quy trình quét sâu...")
    
    for sub in subreddits:
        for keyword in SEARCH_KEYWORDS:
            print(f" -> Đang quét: r/{sub} | Từ khóa: '{keyword}'")
            
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json"
                params = {
                    'q': keyword,
                    'restrict_sr': '1',
                    'limit': 100, # Lấy tối đa mỗi lần
                    'sort': 'relevance', # Lấy bài liên quan nhất thay vì mới nhất
                    't': 'all' # Tìm trong tất cả thời gian
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    print(f"    Lỗi kết nối: {response.status_code}")
                    continue
                    
                data = response.json()
                
                if 'data' not in data or 'children' not in data['data']:
                    continue

                count_added = 0
                for item in data['data']['children']:
                    post = item['data']
                    post_id = post['id']
                    
                    # Gộp tiêu đề và nội dung để kiểm tra
                    full_text = f"{post['title']} {post['selftext']}"
                    
                    # --- BƯỚC LỌC QUAN TRỌNG ---
                    if post_id not in all_posts and strict_filter(full_text):
                        all_posts[post_id] = {
                            'id': post_id,
                            'created_utc': post['created_utc'],
                            'date_readable': pd.to_datetime(post['created_utc'], unit='s'),
                            'title': post['title'],
                            'content': post['selftext'],
                            'full_text': full_text, # Lưu cái này để lát nữa dịch
                            'score': post['score'],
                            'subreddit': sub,
                            'url': post['url']
                        }
                        count_added += 1
                
                print(f"    -> Tìm thấy {len(data['data']['children'])} bài, Lọc được: {count_added} bài chuẩn.")
                
                # Nghỉ tay xíu để Reddit không chặn IP
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"    Lỗi: {e}")

    # Chuyển về DataFrame
    df = pd.DataFrame(list(all_posts.values()))
    return df

# --- CHẠY SCRIPT ---
df_final = crawl_reddit_strict()

# Hiển thị kết quả
if not df_final.empty:
    print("\n" + "="*50)
    print(f"TỔNG KẾT: Đã thu thập được {len(df_final)} bài chất lượng cao.")
    print("="*50)
    print(df_final[['title', 'date_readable']].head(10))
    
    # Lưu file
    df_final.to_csv(".data/voz_data_filtered.csv", index=False, encoding='utf-8-sig')
    print("\nĐã lưu file: .data/voz_data_filtered.csv")
else:
    print("Không tìm thấy bài nào thỏa mãn bộ lọc nghiêm ngặt này.")