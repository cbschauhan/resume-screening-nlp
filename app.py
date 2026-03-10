# requir library
import nltk
import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

nltk.download('punkit')
nltk.download('stopwords')


# Load model
with open("rf_model.pkl", "rb") as file:
    clf = pickle.load(file)
# Load model 
# svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed


def clean_resume(resume_text):
    text = re.sub(r'http\S+\s*', ' ', resume_text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    clean_Text = re.sub(r'\s+', ' ', text)
    return clean_Text

#  web app
def main():


    st.title("Resume Scrining App")

#     # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # uploaded file
        try:
            resume_bytes = uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
            
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')
            
        cleaned_resume= clean_resume(resume_text)
        input_features=tfidf.transform([cleaned_resume])
        if hasattr(input_features, "toarray"):
            input_features = input_features.toarray()

        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)
        
        category_mapping={
            6: 'Data Science',12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }
        
        category_name=category_mapping.get(prediction_id,"Unknow")
        
        st.write("prediction Category: ", category_name)
        

if __name__ == "__main__":
    main()
