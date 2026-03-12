# required libraries
import nltk
import streamlit as st
import pickle
import docx
import PyPDF2
import re
# from Crypto.Cipher import AES





nltk.download('punkt')
nltk.download('stopwords')


# Load model
with open("rf_model.pkl", "rb") as file:
    clf = pickle.load(file)

tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_resume(resume_text):
    text = re.sub(r'http\S+\s*', ' ', resume_text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    clean_Text = re.sub(r'\s+', ' ', text)

    return clean_Text


# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# ---------------------------
# Extract text from DOCX
# ---------------------------
def extract_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# # ---------------------------
# # Extract text from PDF
# # ---------------------------
# def extract_pdf(file):
#     from PyPDF2.errors import DependencyError

#     try:
#         reader = PyPDF2.PdfReader(file)
#     except DependencyError:
#         st.error("PyCryptodome is required for AES-encrypted PDFs. Install with `pip install pycryptodome`.")
#         return ""

#     # Decrypt if encrypted
#     if reader.is_encrypted:
#         try:
#             reader.decrypt("")  # try empty password
#         except Exception as e:
#             st.error(f"Failed to decrypt PDF: {e}")
#             return ""

#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text

#     return text


# ---------------------------
# Keyword Detection
# ---------------------------
def find_resume_keywords(resume_text):

    keywords = [
        "education",
        "experience",
        "work",
        "skills",
        "projects",
        "certifications",
        "certificates",
        "achievements",
        "objective",
        "summary"
    ]

    text = resume_text.lower()

    found_keywords = {}

    for keyword in keywords:
        matches = re.findall(rf"\b{keyword}\b", text)
        found_keywords[keyword] = len(matches)

    matched_count = sum(1 for v in found_keywords.values() if v > 0)

    return found_keywords, matched_count


# ---------------------------
# Main App
# ---------------------------
def main():
    st.title("Resume Screening App")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:

        # Detect file type and extract text properly
        if uploaded_file.type == "application/pdf":
            resume_text = extract_pdf(uploaded_file)

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_docx(uploaded_file)

        else:
            try:
                resume_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = uploaded_file.read().decode("latin-1")

        # Keyword detection
        keyword_matches, matched_count = find_resume_keywords(resume_text)

        if matched_count < 3:
            st.error("❌ Please only upload resume files.")
            return

        st.info(f"✅ Thanks for uploading your resume! Detected {matched_count} relevant sections.")

        cleaned_resume = clean_resume(resume_text)

        input_features = tfidf.transform([cleaned_resume])

        if hasattr(input_features, "toarray"):
            input_features = input_features.toarray()

        prediction_id = clf.predict(input_features)[0]

        st.write("Prediction ID:", prediction_id)

        category_mapping = {
            6: 'Data Science', 12: 'HR',
            0: 'Advocate', 1: 'Arts',
            24: 'Web Designing', 16: 'Mechanical Engineer',
            22: 'Sales', 14: 'Health and fitness',
            5: 'Civil Engineer', 15: 'Java Developer',
            4: 'Business Analyst', 21: 'SAP Developer',
            2: 'Automation Testing', 11: 'Electrical Engineering',
            18: 'Operations Manager', 20: 'Python Developer',
            8: 'DevOps Engineer', 17: 'Network Security Engineer',
            19: 'PMO', 7: 'Database',
            13: 'Hadoop', 10: 'ETL Developer',
            9: 'DotNet Developer', 3: 'Blockchain',
            23: 'Testing'
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.success(f"Prediction Category: {category_name}")


if __name__ == "__main__":
    main()