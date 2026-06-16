import streamlit as st
import re
import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Sumy Framework Modules
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

# ─── PREMIUM DARK SAAS LAYOUT CONFIGURATIONS ───
# ─── PREMIUM DARK SAAS LAYOUT CONFIGURATIONS ───
import streamlit as st
import re
import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Sumy Framework Modules
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

# ─── PREMIUM DARK SAAS LAYOUT CONFIGURATIONS ───
# ─── PREMIUM DARK SAAS LAYOUT CONFIGURATIONS ───
# ─── PREMIUM DARK SAAS LAYOUT CONFIGURATIONS ───
st.set_page_config(
    page_title="Summarizer - Advanced Workspace",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep Injector CSS - Dark Theme Override across all core dashboard assets and multi-column popups
st.markdown("""
    <style>
    /* 🚨 ULTIMATE FIXED BLOCK: Forces deployment choices columns to be dark with visible text 🚨 */
    div[role="dialog"], div[data-testid="stModal"], div[data-testid="stDialog"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
    }
    
    /* Targets the specific white pricing/deployment card containers inside the popup */
    div[role="dialog"] div, div[data-testid="stModal"] div, [data-testid="stHeader"] + div div {
        background-color: #131B2E !important;
        color: #F8FAFC !important;
    }
    
    /* Force checkmarks and descriptive bullet texts to be visible and distinct */
    div[role="dialog"] span, div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3 {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
    }
    
    /* Keep action buttons inside the popup readable */
    div[role="dialog"] button {
        background-color: #1E40AF !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    div[role="dialog"] button:hover {
        background-color: #1D4ED8 !important;
    }

    /* Transparent Top Navigation Bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] div, header[data-testid="stHeader"] a {
        color: #E2E8F0 !important;
    }

    /* Main Background & Grid Overlay */
    .stApp {
        background-color: #0B0F19 !important;
        background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    
    /* Global Text Modifications */
    h1, h2, h3, h4, p, label, span { color: #E2E8F0 !important; font-family: 'Inter', sans-serif; }
    
    /* Header Styles */
    .header-container { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding-top: 10px; }
    .header-logo { background-color: #0E7490; padding: 8px; border-radius: 8px; font-weight: bold; }
    .header-subtitle { font-size: 0.85rem; color: #94A3B8 !important; margin-top: -4px; font-weight: 500; }

    /* SaaS Dark Container Blocks */
    .saas-card {
        background-color: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        min-height: 150px;
    }
    .card-header { 
        font-size: 0.9rem; font-weight: 600; text-transform: uppercase; 
        letter-spacing: 0.05em; color: #38BDF8 !important; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
    }
    
    /* Sidebar / History Section placeholder */
    .history-box {
        border: 2px dashed #1E293B; border-radius: 10px; padding: 2rem 1rem;
        text-align: center; color: #475569 !important; font-size: 0.9rem;
    }

    /* Primary Generate Action Button Customization */
    .stButton>button {
        background-color: #1E40AF !important; color: white !important;
        border: none !important; border-radius: 8px !important; width: 100%;
        padding: 12px !important; font-weight: 600 !important; transition: background 0.2s ease;
    }
    .stButton>button:hover { background-color: #1D4ED8 !important; }

    /* HIGH CONTRAST INPUT OVERRIDE FIX */
    div[data-baseweb="textarea"] { 
        background-color: #0F172A !important; 
        border: 1px solid #334155 !important; 
        border-radius: 8px !important; 
    }
    textarea { 
        color: #FFFFFF !important; 
        -webkit-text-fill-color: #FFFFFF !important; 
        background-color: #0F172A !important;
    }
    
    /* Metrics block formatting */
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #38BDF8; }
    .metric-lbl { font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def initialize_nlp():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

initialize_nlp()

class HybridSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        # 🔑 Active Hugging Face API access token string
        self.token = "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" 
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def clean_text(self, text):
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def native_tfidf(self, raw_text, num_sentences):
        cleaned = self.clean_text(raw_text)
        sentences = sent_tokenize(cleaned)
        if len(sentences) <= num_sentences:
            return cleaned
        vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        ranked_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        return " ".join([sentences[idx] for idx in sorted(ranked_indices)])

    def sumy_lexrank(self, raw_text, num_sentences):
        cleaned = self.clean_text(raw_text)
        parser = PlaintextParser.from_string(cleaned, Tokenizer('english'))
        summarizer = LexRankSummarizer()
        return " ".join([str(s) for s in summarizer(parser.document, num_sentences)])

    def sumy_luhn(self, raw_text, num_sentences):
        cleaned = self.clean_text(raw_text)
        parser = PlaintextParser.from_string(cleaned, Tokenizer('english'))
        summarizer = LuhnSummarizer()
        return " ".join([str(s) for s in summarizer(parser.document, num_sentences)])

    def cloud_abstractive_chunked(self, raw_text, max_len):
        cleaned = self.clean_text(raw_text)
        sentences = sent_tokenize(cleaned)
        chunks, current_chunk, current_words = [], [], 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if current_words + word_count > 400:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_words = word_count
            else:
                current_chunk.append(sentence)
                current_words += word_count
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        summarized_chunks = []
        for chunk in chunks:
            payload = {"inputs": chunk, "parameters": {"max_length": int(max_len), "min_length": 30}}
            try:
                response = requests.post(self.API_URL, headers=self.headers, json=payload, timeout=30)
                output = response.json()
                if isinstance(output, dict) and "estimated_time" in output:
                    import time
                    time.sleep(10)
                    response = requests.post(self.API_URL, headers=self.headers, json=payload, timeout=30)
                    output = response.json()
                summarized_chunks.append(output[0]['summary_text'])
            except Exception:
                summarized_chunks.append(self.native_tfidf(chunk, num_sentences=2))
        return " ".join(summarized_chunks)


# ─── BRANDING HEADER CONTAINER ───
st.markdown("""
    <div class='header-container'>
        <div class='header-logo'>📖</div>
        <div>
            <div style='font-size:1.4rem; font-weight:700; color:#FFFFFF;'>summarizer</div>
            <div class='header-subtitle'>NLTK / Scikit-learn / Sumy Core Core Deployment Pipeline</div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ─── CORE 3-COLUMN WORKSPACE INTERFACE ───
left_sidebar, center_input, right_output = st.columns([1, 1.3, 1.5], gap="medium")

with left_sidebar:
    st.markdown("<div class='card-header'>🕒 History</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='history-box'>
            <div style='font-size:1.5rem; margin-bottom:0.5rem;'>📄</div>
            <strong>No saved summaries yet</strong><br>
            <span style='font-size:0.8rem; color:#475569;'>Generate and save a summary to see it here</span>
        </div>
    """, unsafe_allow_html=True)

with center_input:
    st.markdown("<div class='card-header'>📄 Input Text</div>", unsafe_allow_html=True)
    input_corpus = st.text_area("", height=280, placeholder="Paste or type the text you want to summarize...", label_visibility="collapsed")
    
    # Generate action button
    process_btn = st.button("Generate Summary")
    
    st.write("")
    st.markdown("<div class='card-header'>⚙️ Algorithm Selection</div>", unsafe_allow_html=True)
    algorithm = st.selectbox(
        "Select Active Operational Engine",
        (
            "✨ Cloud Neural Network (Abstractive AI)", 
            "Native TF-IDF (Scikit-Learn Model)", 
            "LexRank Engine (Graph-Based)", 
            "Luhn Engine (Frequency-Based)"
        ),
        label_visibility="collapsed"
    )
    
    # Conditional configurations sidebar slider style simulation
    if "Abstractive AI" in algorithm:
        max_len = st.slider("Target Word Length", min_value=30, max_value=150, value=75)
    else:
        summary_length = st.slider("Sentence Constraint Count", min_value=1, max_value=10, value=3)

with right_output:
    st.markdown("<div class='card-header'>📥 Optimized Summary Viewport</div>", unsafe_allow_html=True)
    
    if process_btn and input_corpus.strip():
        engine = HybridSummarizer()
        input_words = len(input_corpus.split())
        
        with st.spinner("Processing natural language telemetry logic..."):
            if "Abstractive AI" in algorithm:
                result = engine.cloud_abstractive_chunked(input_corpus, max_len)
            elif "TF-IDF" in algorithm:
                result = engine.native_tfidf(input_corpus, summary_length)
            elif "LexRank" in algorithm:
                result = engine.sumy_lexrank(input_corpus, summary_length)
            else:
                result = engine.sumy_luhn(input_corpus, summary_length)
                
        output_words = len(result.split())
        compression_ratio = round((1 - (output_words / input_words)) * 100, 1)
        
        # Display Summary block mirroring the SaaS look
        st.markdown(f"""
            <div style='background-color:#1E293B; border: 1px solid #334155; border-radius:10px; padding:1.5rem; color:#F1F5F9; font-size:1rem; line-height:1.6; min-height:280px;'>
                {result}
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row cards layout
        st.write("")
        st.markdown("<div class='card-header'>📊 Operational Telemetry Metrics</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='saas-card' style='min-height:90px; text-align:center; padding:12px;'><div class='metric-val'>{input_words}</div><div class='metric-lbl'>Source Words</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='saas-card' style='min-height:90px; text-align:center; padding:12px;'><div class='metric-val'>{output_words}</div><div class='metric-lbl'>Target Words</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='saas-card' style='min-height:90px; text-align:center; padding:12px;'><div class='metric-val'>-{compression_ratio}%</div><div class='metric-lbl'>Reduction Footprint</div></div>", unsafe_allow_html=True)
            
    else:
        st.markdown("""
            <div style='border: 2px dashed #1E293B; border-radius:12px; padding:6.5rem 1rem; text-align:center; color:#475569; min-height:280px;'>
                <div style='font-size:2rem; margin-bottom:0.5rem;'>📥</div>
                <strong>Summary will appear here</strong><br>
                <span style='font-size:0.8rem;'>Enter text payload data and click generate</span>
            </div>
        """, unsafe_allow_html=True)
