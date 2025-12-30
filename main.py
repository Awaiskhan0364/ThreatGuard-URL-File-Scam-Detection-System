from flask import Flask, render_template, request
import PyPDF2

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# =====================================================
# TRAINING DATA (DEMO DATASET – FOR PRESENTATION)
# =====================================================

# -------- FILE SCAM DATA --------
file_texts = [
    "you have won a lottery claim your prize now",
    "this is a normal business email meeting tomorrow",
    "urgent your bank account is compromised",
    "please find attached the project report",
    "fraud alert verify your identity immediately",
    "invoice for your recent purchase"
]

file_labels = [
    "scam",
    "safe",
    "scam",
    "safe",
    "scam",
    "safe"
]

# -------- URL THREAT DATA --------
url_texts = [
    "http://secure-login-paypal.com",
    "https://google.com",
    "http://phishingsite.xyz/login",
    "https://github.com",
    "http://malware-download.exe",
    "http://deface-attack-site.com"
]

url_labels = [
    "phishing",
    "benign",
    "phishing",
    "benign",
    "malware",
    "defacement"
]

# =====================================================
# SEPARATE VECTORIZERS & MODELS
# =====================================================

file_vectorizer = TfidfVectorizer()
url_vectorizer = TfidfVectorizer()

# Train file model
X_file = file_vectorizer.fit_transform(file_texts)
file_model = MultinomialNB()
file_model.fit(X_file, file_labels)

# Train URL model
X_url = url_vectorizer.fit_transform(url_texts)
url_model = MultinomialNB()
url_model.fit(X_url, url_labels)

# =====================================================
# ROUTES
# =====================================================

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        message="",
        predicted_class="",
        input_url=""
    )

# ---------------- FILE SCAN ----------------
@app.route("/scam/", methods=["POST"])
def detect_scam():
    if "file" not in request.files:
        return render_template(
            "index.html",
            message="❌ No file uploaded",
            predicted_class="",
            input_url=""
        )

    file = request.files["file"]
    text = ""

    # TXT file
    if file.filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")

    # PDF file
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

    else:
        return render_template(
            "index.html",
            message="❌ Unsupported file format",
            predicted_class="",
            input_url=""
        )

    if not text.strip():
        return render_template(
            "index.html",
            message="❌ No readable text found",
            predicted_class="",
            input_url=""
        )

    # ML Prediction
    vector = file_vectorizer.transform([text])
    prediction = file_model.predict(vector)[0]

    message = "⚠️ Scam / Fake content detected" if prediction == "scam" else "✅ File content looks safe"

    return render_template(
        "index.html",
        message=message,
        predicted_class=prediction,
        input_url=""
    )

# ---------------- URL SCAN ----------------
@app.route("/predict", methods=["POST"])
def detect_url():
    url = request.form.get("url", "").strip()
    if not url:
        return render_template(
            "index.html",
            message="❌ No URL provided",
            predicted_class="",
            input_url=""
        )

    vector = url_vectorizer.transform([url])
    prediction = url_model.predict(vector)[0]

    if prediction == "benign":
        message = "✅ URL looks safe"
    elif prediction == "phishing":
        message = "⚠️ Phishing URL detected"
    elif prediction == "malware":
        message = "⚠️ Malware URL detected"
    elif prediction == "defacement":
        message = "⚠️ Defacement / Attack URL detected"
    else:
        message = "❌ Unknown threat"

    return render_template(
        "index.html",
        message=message,
        predicted_class=prediction,
        input_url=url
    )

# =====================================================
if __name__ == "__main__":
    print("Starting ThreatGuard server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
