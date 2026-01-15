import pandas as pd
import requests
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import logging
from deep_translator import GoogleTranslator
import os

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLTK SETUP ---
# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:    
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# --- PREPROCESSING FUNCTIONS (From main.ipynb) ---
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # 1. Cleaning
    text = str(text).encode("ascii", "ignore").decode()
    text = re.sub(r"https?:\/\/\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.lower()

    # 2. Tokenization
    pattern = r"\w+(?:'\w+)?|[^\w\s]+"
    tokens = re.findall(pattern, text)

    # 3. Batch POS Tagging (Faster than tagging word-by-word)
    tagged_tokens = nltk.pos_tag(tokens)

    # 4. Lemmatization with POS context
    clean_tokens = []
    for word, tag in tagged_tokens:
        if re.match(r"\w+", word):
            pos = get_wordnet_pos(tag)
            clean_tokens.append(lemmatizer.lemmatize(word, pos))
        else:
            clean_tokens.append(word)

    return " ".join(clean_tokens)

custom_stop_words = [
    "to", "the", "and", "of", "in", "for", "with", "on", "that", "this",
    "it", "is", "be", "wa", "so", "but", "or", "as", "at", "by",
]

def remove_stop_words(text):
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t not in custom_stop_words]
    return " ".join(filtered_tokens)

def full_preprocess(text):
    """Combines all preprocessing steps"""
    processed = preprocess_text(text)
    processed = remove_stop_words(processed)
    return processed

# --- CRAWL FUNCTIONS (Adapted from app_v2/crawl.py) ---
SEARCH_KEYWORDS = ["thi", "đồ án", "nợ môn", "học lại", "ra trường", "áp lực học", "rớt môn"]
ACADEMIC_KEYWORDS = [
    "thi cuối kỳ", "thi giữa kỳ", "thi lại", "trả nợ môn", "học cải thiện",
    "thi thpt", "thi đại học", "xét tuyển", "nguyện vọng", "học bạ",
    "điểm thi", "điểm chuẩn", "đề thi", "ôn thi", "luyện thi",
    "khối a", "khối b", "khối c", "khối d", "đánh giá năng lực", "đgnl",
    "đồ án tốt nghiệp", "khóa luận", "deadline dí",
    "rớt môn", "tạch môn", "nợ môn", "bảng điểm", "gpa", "cpa",
    "cảnh cáo học vụ", "bị đuổi học", "bảo lưu", "đăng ký tín chỉ",
    "ra trường", "thất nghiệp", "nộp bài", "phúc khảo", "qua môn",
    "áp lực đồng trang lứa", "peer pressure", "stress vì học",
    "reset", "đăng xuất", "tự tử", "nhảy lầu", "bế tắc",
    "mất gốc", "học kém", "ngu học", "chán học", "áp lực điểm số"
]
BLACKLIST_KEYWORDS = [
    "tuyển dụng", "việc làm", "shopee", "lazada", "tiki", "thanh lý",
    "pass lại", "voucher", "nhượng phòng", "tìm trọ", "thi công", 
    "thiết kế", "xây dựng", "thi hoa hậu", "sex", "kèo bóng",
    "bằng lái", "sát hạch", "b2", "a1", "lái xe", "gplx"
]

def strict_filter(text):
    if not isinstance(text, str): return False
    text_lower = text.lower()
    
    if len(text_lower) < 30: return False
    
    for bad_word in BLACKLIST_KEYWORDS:
        if bad_word in text_lower: return False
        
    is_academic = False
    for kw in ACADEMIC_KEYWORDS:
        if kw in text_lower:
            is_academic = True
            break
            
    return is_academic

def crawl_reddit_live(limit=20):
    """
    Crawls a small number of posts for live demo.
    """
    all_posts = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # We just search one random keyword to be fast
    import random
    keyword = random.choice(SEARCH_KEYWORDS)
    sub = "vozforums"
    
    logger.info(f"Crawling live data for keyword: {keyword}")
    
    try:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {
            'q': keyword,
            'restrict_sr': '1',
            'limit': limit, 
            'sort': 'new', # Get NEWEST posts
            't': 'all'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'children' in data['data']:
                for item in data['data']['children']:
                    post = item['data']
                    full_text = f"{post['title']} {post['selftext']}"
                    
                    if strict_filter(full_text):
                        all_posts.append({
                            'id': post['id'],
                            'created_utc': post['created_utc'],
                            'date_readable': pd.to_datetime(post['created_utc'], unit='s'),
                            'full_text': full_text
                        })
                        if len(all_posts) >= 5: # Limit detailed processing for speed
                            break
    except Exception as e:
        logger.error(f"Error crawling: {e}")
        
    return pd.DataFrame(all_posts)

# --- TRANSLATION FUNCTION ---
# Simple Mapping for Slang (from app_v2/translate.py)
SLANG_MAPPING = {
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
    if not isinstance(text, str): return ""
    text_lower = text.lower()
    for vn_word, en_word in SLANG_MAPPING.items():
        if vn_word in text_lower:
            text_lower = text_lower.replace(vn_word, en_word)
    return text_lower

def translate_text(text):
    try:
        # Pre-process slang map
        text_mapped = map_vietnamese_slang(text)
        
        # Translate
        # Split if too long (simple chunking)
        if len(text_mapped) > 4500:
            text_mapped = text_mapped[:4500]
            
        translated = GoogleTranslator(source='vi', target='en').translate(text_mapped)
        return translated
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text # Return original if fail
