import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from collections import Counter

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CyberGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; color: white; }
    .main-header p  { font-size: 1.1rem; margin: 0.5rem 0 0; color: #a0aec0; }

    .module-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .result-safe {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .result-danger {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
        color: #721c24;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .result-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box h3 { margin: 0; font-size: 1.8rem; color: #2c3e50; }
    .metric-box p  { margin: 0; font-size: 0.85rem; color: #6c757d; }
    .chat-user {
        background: #e3f2fd;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        color: #1a237e;
    }
    .chat-bot {
        background: #f3e5f5;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        color: #4a148c;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ CyberGuard AI</h1>
    <p>AI-Powered Cybersecurity Assistant — Phishing Detection • Password Analysis • Security Chatbot</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your key from console.groq.com"
    )

    st.markdown("---")
    st.markdown("### 📊 Project Info")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-box"><h3>96.7%</h3><p>URL Detection</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><h3>99.9%</h3><p>Password AI</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Models Used")
    st.markdown("- 🌲 Random Forest (URLs)")
    st.markdown("- 🧬 Neural Network (Passwords)")
    st.markdown("- 🤖 LLaMA 3.3 70B (Chatbot)")
    st.markdown("---")
    st.caption("Information Security Project")

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained models — same as your notebook"""
    try:
        import tldextract
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        # ── Phishing Dataset & Model ──
        df = pd.read_csv('dataset_phishing.csv')
        df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
        feature_cols = [c for c in df.columns if c not in ['url','status','label']]
        X = df[feature_cols]
        y = df['label']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        rf_model = RandomForestClassifier(
            n_estimators=200, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        legit_means = df[df['status']=='legitimate'][feature_cols].mean()

        # ── Password Dataset & Model ──
        df_pass = pd.read_csv('data.csv', on_bad_lines='skip').dropna()
        df_pass['password'] = df_pass['password'].astype(str)
        df_pass['strength'] = df_pass['strength'].astype(int)
        df_pass = df_pass.groupby('strength', group_keys=False).apply(
            lambda x: x.sample(min(5000, len(x)), random_state=42)
        ).reset_index(drop=True)

        def extract_features(password):
            password = str(password)
            f = {}
            f['length']       = len(password)
            f['has_upper']    = 1 if re.search(r'[A-Z]', password) else 0
            f['has_lower']    = 1 if re.search(r'[a-z]', password) else 0
            f['has_digit']    = 1 if re.search(r'\d', password) else 0
            f['has_special']  = 1 if re.search(r'[!@#$%^&*()\[\],.?":{}|<>_\-]', password) else 0
            f['num_upper']    = sum(c.isupper() for c in password)
            f['num_lower']    = sum(c.islower() for c in password)
            f['num_digits']   = sum(c.isdigit() for c in password)
            f['num_special']  = sum(not c.isalnum() for c in password)
            f['unique_chars'] = len(set(password))
            n = max(len(password), 1)
            f['char_diversity'] = len(set(password)) / n
            f['type_count']   = f['has_upper']+f['has_lower']+f['has_digit']+f['has_special']
            f['ratio_upper']  = f['num_upper'] / n
            f['ratio_lower']  = f['num_lower'] / n
            f['ratio_digits'] = f['num_digits'] / n
            f['ratio_special']= f['num_special'] / n
            counts  = Counter(password)
            f['entropy'] = -sum((v/n)*np.log2(v/n) for v in counts.values())
            common = ['123456','password','qwerty','abc123','admin','letmein',
                      'welcome','monkey','dragon','sunshine','princess','iloveyou',
                      '111111','12345678','123123','football','shadow','master']
            f['is_common']     = 1 if password.lower() in common else 0
            f['has_seq_num']   = 1 if any(str(i)+str(i+1)+str(i+2) in password for i in range(8)) else 0
            f['has_seq_alpha'] = 1 if any(chr(i)+chr(i+1)+chr(i+2) in password.lower()
                                          for i in range(ord('a'), ord('x'))) else 0
            f['has_repeat']    = 1 if re.search(r'(.)\1{2,}', password) else 0
            score = 0
            if f['length']>=8: score+=1
            if f['length']>=12: score+=1
            if f['has_upper']: score+=1
            if f['has_lower']: score+=1
            if f['has_digit']: score+=1
            if f['has_special']: score+=1
            if f['char_diversity']>0.7: score+=1
            if f['entropy']>3: score+=1
            f['heuristic_score'] = score
            return f

        feat_list = [extract_features(p) for p in df_pass['password']]
        feat_df   = pd.DataFrame(feat_list)
        X_p = feat_df
        y_p = df_pass['strength'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_p, test_size=0.2, random_state=42, stratify=y_p)

        nn_model = MLPClassifier(
            hidden_layer_sizes=(128,64,32), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        nn_model.fit(X_tr, y_tr)

        return {
            'rf_model': rf_model,
            'feature_cols': feature_cols,
            'legit_means': legit_means,
            'nn_model': nn_model,
            'scaler': scaler,
            'extract_features': extract_features,
            'pass_feature_cols': list(feat_df.columns),
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}

# ── URL Prediction ────────────────────────────────────────────
def predict_url(url, models):
    import tldextract
    extracted = tldextract.extract(url)
    domain    = extracted.domain
    subdomain = extracted.subdomain
    suffix    = extracted.suffix
    parts     = url.split('/')
    path      = '/'.join(parts[3:]) if len(parts) > 3 else ''
    fc        = models['feature_cols']
    lm        = models['legit_means']

    features = {col: lm.get(col, 0) for col in fc}

    features['length_url']        = len(url)
    features['length_hostname']   = len(domain+'.'+suffix)
    features['ip']                = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    features['nb_dots']           = url.count('.')
    features['nb_hyphens']        = url.count('-')
    features['nb_at']             = url.count('@')
    features['nb_qm']             = url.count('?')
    features['nb_and']            = url.count('&')
    features['nb_or']             = url.count('|')
    features['nb_eq']             = url.count('=')
    features['nb_underscore']     = url.count('_')
    features['nb_tilde']          = url.count('~')
    features['nb_percent']        = url.count('%')
    features['nb_slash']          = url.count('/')
    features['nb_star']           = url.count('*')
    features['nb_colon']          = url.count(':')
    features['nb_comma']          = url.count(',')
    features['nb_semicolumn']     = url.count(';')
    features['nb_dollar']         = url.count('$')
    features['nb_space']          = url.count(' ')
    features['nb_www']            = 1 if 'www.' in url.lower() else 0
    features['nb_com']            = 1 if '.com' in url.lower() else 0
    features['nb_dslash']         = url.count('//')
    features['http_in_path']      = 1 if 'http' in path.lower() else 0
    features['https_token']       = 1 if url.startswith('https') else 0
    features['ratio_digits_url']  = sum(c.isdigit() for c in url) / max(len(url),1)
    features['ratio_digits_host'] = sum(c.isdigit() for c in domain) / max(len(domain),1)
    features['punycode']          = 1 if 'xn--' in url.lower() else 0
    features['port']              = 1 if re.search(r':\d{2,5}/', url) else 0
    features['tld_in_path']       = 1 if suffix in path.lower() else 0
    features['tld_in_subdomain']  = 1 if suffix in subdomain.lower() else 0
    features['abnormal_subdomain']= 1 if re.search(r'(w[0-9]+|mail\d)', subdomain) else 0
    features['nb_subdomains']     = len(subdomain.split('.')) if subdomain else 0
    features['prefix_suffix']     = 1 if '-' in domain else 0
    features['random_domain']     = 1 if re.search(r'[0-9]{4,}', domain) else 0
    features['shortening_service']= 1 if any(s in url for s in [
        'bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd']) else 0
    features['path_extension']    = 1 if re.search(r'\.(exe|zip|php|html)$', path.lower()) else 0
    features['nb_redirection']    = url.count('//')-1 if '//' in url else 0
    features['nb_external_redirection'] = 0

    all_words  = [w for w in re.split(r'\W+', url) if w]
    host_words = [w for w in re.split(r'[\.\-]', domain) if w]
    path_words = [w for w in re.split(r'\W+', path) if w]
    features['length_words_raw']   = len(all_words)
    features['char_repeat']        = max((url.count(c) for c in set(url)), default=0)
    features['shortest_words_raw'] = min((len(w) for w in all_words), default=0)
    features['shortest_word_host'] = min((len(w) for w in host_words), default=0)
    features['shortest_word_path'] = min((len(w) for w in path_words), default=0)
    features['longest_words_raw']  = max((len(w) for w in all_words), default=0)
    features['longest_word_host']  = max((len(w) for w in host_words), default=0)
    features['longest_word_path']  = max((len(w) for w in path_words), default=0)
    features['avg_words_raw']      = np.mean([len(w) for w in all_words]) if all_words else 0
    features['avg_word_host']      = np.mean([len(w) for w in host_words]) if host_words else 0
    features['avg_word_path']      = np.mean([len(w) for w in path_words]) if path_words else 0
    features['phish_hints']        = 1 if any(w in url.lower() for w in [
        'login','verify','secure','account','update','confirm','banking',
        'prize','winner','free','password','suspend','alert','validate']) else 0
    features['domain_in_brand']    = 1 if any(b in domain.lower() for b in [
        'google','paypal','amazon','microsoft','apple','facebook','netflix']) else 0
    features['brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in [
        'google','paypal','amazon','microsoft','apple','facebook']) else 0
    features['brand_in_path']      = 1 if any(b in path.lower() for b in [
        'google','paypal','amazon','microsoft','apple','facebook']) else 0
    features['suspecious_tld']     = 1 if suffix in [
        'tk','ml','ga','cf','gq','xyz','top','click','link','pw'] else 0
    features['statistical_report'] = 0

    if features['phish_hints'] or features['suspecious_tld'] or features['ip']:
        features['page_rank'] = 0
        features['web_traffic'] = 0
        features['domain_age'] = 0
        features['domain_registration_length'] = 0
        features['google_index'] = 1
        features['dns_record'] = 0
    if features['https_token']:
        features['page_rank'] = max(features.get('page_rank', 0), 2)

    inp        = pd.DataFrame([features])[fc]
    prediction = models['rf_model'].predict(inp)[0]
    proba      = models['rf_model'].predict_proba(inp)[0]
    return prediction, proba

# ── Password Prediction ───────────────────────────────────────
def predict_password(password, models):
    features       = models['extract_features'](password)
    feat_input     = pd.DataFrame([features])[models['pass_feature_cols']]
    feat_scaled    = models['scaler'].transform(feat_input)
    prediction     = models['nn_model'].predict(feat_scaled)[0]
    proba          = models['nn_model'].predict_proba(feat_scaled)[0]
    return prediction, proba, features

# ── Chatbot ───────────────────────────────────────────────────
def ask_chatbot(user_message, history, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    system = """You are CyberGuard AI, an expert cybersecurity assistant.
Your expertise: phishing, passwords, malware, network security, data privacy, social engineering.
Guidelines: clear practical advice, bullet points, emojis, max 300 words, end with a quick tip."""
    messages = [{"role": "system", "content": system}]
    for h in history:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_message})
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Load models
with st.spinner("🔄 Loading AI models... (first time takes ~1-2 min)"):
    models = load_models()

if not models['loaded']:
    st.error(f"❌ Models load nahi hue: {models.get('error','Unknown error')}")
    st.info("💡 Make sure `dataset_phishing.csv` and `data.csv` same folder mein hain.")
    st.stop()

st.success("✅ All AI Models Loaded Successfully!")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 Phishing Detector",
    "🔐 Password Analyzer",
    "🤖 Security Chatbot",
    "📊 Project Summary"
])

# ════════════════════════════════════════════
# TAB 1 — PHISHING URL DETECTOR
# ════════════════════════════════════════════
with tab1:
    st.markdown("## 🔗 Phishing URL Detector")
    st.markdown("Enter a URL to check if it's safe or a phishing attempt.")

    url_input = st.text_input(
        "🌐 Enter URL",
        placeholder="https://www.example.com",
        key="url_input"
    )

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔍 Analyze URL", key="analyze_url"):
            if url_input:
                with st.spinner("Analyzing..."):
                    pred, proba = predict_url(url_input, models)
                risk = proba[1]*100
                safe = proba[0]*100
                if pred == 1:
                    st.markdown(f'<div class="result-danger">🚨 PHISHING URL DETECTED!<br>Risk Score: {risk:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe">✅ LEGITIMATE URL<br>Safe Score: {safe:.1f}%</div>', unsafe_allow_html=True)

                st.markdown("### 📊 Confidence Scores")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🛡️ Safe Score",  f"{safe:.1f}%")
                    st.progress(safe/100)
                with c2:
                    st.metric("⚠️ Risk Score",  f"{risk:.1f}%")
                    st.progress(risk/100)
            else:
                st.warning("⚠️ Please enter a URL first!")

    st.markdown("### 🧪 Quick Test Examples")
    examples = [
        ("✅ Google",  "https://www.google.com"),
        ("✅ Amazon",  "https://www.amazon.com"),
        ("🚨 Phishing","http://google-security-alert.com/login/verify"),
        ("🚨 Fake",    "http://free-prize-winner.tk/claim-now"),
    ]
    cols = st.columns(4)
    for i, (label, url) in enumerate(examples):
        with cols[i]:
            if st.button(label, key=f"ex_url_{i}"):
                with st.spinner("Analyzing..."):
                    pred, proba = predict_url(url, models)
                risk = proba[1]*100
                safe = proba[0]*100
                if pred == 1:
                    st.markdown(f'<div class="result-danger">🚨 Phishing!<br>{risk:.1f}% risk</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe">✅ Safe!<br>{safe:.1f}% safe</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — PASSWORD ANALYZER
# ════════════════════════════════════════════
with tab2:
    st.markdown("## 🔐 Password Strength Analyzer")
    st.markdown("Check how strong your password is using Neural Network AI.")

    pass_input = st.text_input(
        "🔑 Enter Password",
        type="password",
        placeholder="Enter your password...",
        key="pass_input"
    )

    show_pass = st.checkbox("👁️ Show password", key="show_pass")
    if show_pass and pass_input:
        st.code(pass_input)

    if st.button("🔍 Analyze Password", key="analyze_pass"):
        if pass_input:
            with st.spinner("Analyzing..."):
                pred, proba, feats = predict_password(pass_input, models)

            labels = {0: ("🔴 WEAK",   "result-danger"),
                      1: ("🟡 MEDIUM", "result-warning"),
                      2: ("🟢 STRONG", "result-safe")}
            label, css = labels[pred]
            conf = proba[pred]*100

            st.markdown(f'<div class="{css}">💪 Strength: {label} &nbsp;|&nbsp; Confidence: {conf:.1f}%</div>',
                        unsafe_allow_html=True)

            st.markdown("### 📋 Detailed Analysis")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📏 Length",    f"{feats['length']} chars")
            c2.metric("🔣 Types",     f"{feats['type_count']}/4")
            c3.metric("🎲 Entropy",   f"{feats['entropy']:.2f}")
            c4.metric("🔄 Unique",    f"{feats['unique_chars']}")

            st.markdown("### ✅ Character Checks")
            ch1, ch2, ch3, ch4 = st.columns(4)
            ch1.markdown(f"{'✅' if feats['has_upper'] else '❌'} Uppercase")
            ch2.markdown(f"{'✅' if feats['has_lower'] else '❌'} Lowercase")
            ch3.markdown(f"{'✅' if feats['has_digit'] else '❌'} Numbers")
            ch4.markdown(f"{'✅' if feats['has_special'] else '❌'} Special")

            suggestions = []
            if feats['length'] < 8:   suggestions.append("➕ Use at least 8 characters")
            elif feats['length'] < 12: suggestions.append("➕ Use 12+ characters for better security")
            if not feats['has_upper']:  suggestions.append("➕ Add uppercase letters (A-Z)")
            if not feats['has_lower']:  suggestions.append("➕ Add lowercase letters (a-z)")
            if not feats['has_digit']:  suggestions.append("➕ Add numbers (0-9)")
            if not feats['has_special']:suggestions.append("➕ Add special characters (!@#$%)")
            if feats['is_common']:      suggestions.append("🚫 This is a very common password!")
            if feats['has_seq_num']:    suggestions.append("⚠️ Avoid sequential numbers (123)")
            if feats['has_seq_alpha']:  suggestions.append("⚠️ Avoid sequential letters (abc)")
            if feats['has_repeat']:     suggestions.append("⚠️ Avoid repeated chars (aaa)")

            if suggestions:
                st.markdown("### 💡 Suggestions")
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.success("✅ Perfect password! No improvements needed.")
        else:
            st.warning("⚠️ Please enter a password first!")

# ════════════════════════════════════════════
# TAB 3 — AI SECURITY CHATBOT
# ════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 CyberGuard AI — Security Chatbot")
    st.markdown("Ask me anything about cybersecurity!")

    if not groq_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use the chatbot.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display history
        for msg in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🤖 <b>CyberGuard AI:</b><br>{msg["assistant"]}</div>', unsafe_allow_html=True)

        # Input
        user_msg = st.text_input("💬 Your question:", placeholder="What is phishing?", key="chat_input")

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Send 📤", key="send_chat"):
                if user_msg:
                    with st.spinner("CyberGuard AI is thinking..."):
                        reply = ask_chatbot(user_msg, st.session_state.chat_history, groq_key)
                    st.session_state.chat_history.append({
                        "user": user_msg,
                        "assistant": reply
                    })
                    st.rerun()
        with c2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        # Quick questions
        st.markdown("### 💡 Quick Questions")
        qq = [
            "What is phishing?",
            "How to create strong password?",
            "What is ransomware?",
            "How to enable 2FA?",
            "What is a VPN?"
        ]
        cols = st.columns(5)
        for i, q in enumerate(qq):
            with cols[i]:
                if st.button(q, key=f"qq_{i}"):
                    with st.spinner("Thinking..."):
                        reply = ask_chatbot(q, st.session_state.chat_history, groq_key)
                    st.session_state.chat_history.append({
                        "user": q, "assistant": reply})
                    st.rerun()

# ════════════════════════════════════════════
# TAB 4 — PROJECT SUMMARY
# ════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Project Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔗 URL Accuracy",  "96.72%", "Random Forest")
    c2.metric("🔐 Pass Accuracy", "99.87%", "Neural Network")
    c3.metric("🤖 LLM Model",     "70B",    "LLaMA 3.3")
    c4.metric("📦 Total Modules", "3",      "Integrated")

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🔗 Module 1: Phishing Detection**
        - Model: Random Forest (200 trees)
        - Dataset: 11,430 URLs
        - Features: 87 URL features
        - Accuracy: **96.72%**
        - CV Score: 5-fold validated
        """)
    with col2:
        st.markdown("""
        **🔐 Module 2: Password Analyzer**
        - Model: MLP Neural Network
        - Dataset: 15,000 passwords (Kaggle)
        - Features: 22 engineered features
        - Accuracy: **99.87%**
        - Classes: Weak / Medium / Strong
        """)
    with col3:
        st.markdown("""
        **🤖 Module 3: AI Chatbot**
        - Model: LLaMA 3.3 70B via Groq
        - Multi-turn conversation
        - Real-time security advice
        - Topics: 7+ cybersecurity areas
        - Response: < 2 seconds
        """)

    st.markdown("---")
    st.markdown("### 🛠️ Technologies Used")
    tech_cols = st.columns(5)
    techs = ["Python 3", "Scikit-learn", "Groq API", "Streamlit", "Pandas/NumPy"]
    for i, t in enumerate(techs):
        tech_cols[i].success(t)

    st.markdown("---")
    st.info("🎓 Information Security Course Project | AI-Powered Cybersecurity Assistant")
