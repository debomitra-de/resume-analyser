#streamlit run resume/resume_analyser.py --server.enableXsrfProtection false
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os
from langchain_community.llms import Ollama
import json
from langchain.chains import RetrievalQA
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_job_descriptions(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

def description(data):
  job_title_description = {}  
  for job in data:
    job_title = job['job_title']
    desc = job['job_title'] + " " +  " ".join(job['minimum_qualifications']) + " ".join(job['preferred_qualifications'])
    job_title_description[job_title] = desc
  return job_title_description
json_file_path = os.path.join(working_dir, 'job_descriptions.json')
#get job titles
job_titles = [job['job_title'] for job in load_job_descriptions(json_file_path)]
# Load job descriptions
job_descriptions_data = load_job_descriptions(json_file_path)
# Get the job and description dictionary
job_and_description = description(job_descriptions_data)
def check_extension(file_path):
  _, ext = os.path.splitext(file_path)
  return ext

def process_resume(uploaded_file):
  try:
    if str.lower(check_extension(file_path)) == "pdf":
      loader = PyPDFLoader(file_path)
      documents = loader.load()
      text = documents[0].page_content
      return text
    elif str.lower(check_extension(file_path)) == str.lower("docx"):
      loader = UnstructuredWordDocumentLoader(file_path)
      documents = loader.load()
      text = documents[0].page_content
      return text
  except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
  

# Create embeddings and vector store
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc))

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("Analyse Applicants Resume")

uploaded_file = st.file_uploader("Upload your PDF or DOCX file", type=["pdf", "docx"])

job = st.selectbox("Select the job title", job_titles, index = None, placeholder = "Choose a Job Title")

llm = Ollama(
    model = "llama3:latest",
    temperature = 0.7,
    system = f"You are a HR Assistant. Given the resume of the applicant and job description of the selected job title {job} answer the given prompt. Write Score first followed by hits and miss. Be a little strict in your assessments and scoring."
)

if st.button("Get Answer"):
    file_name = uploaded_file.name
    file_path = os.path.join(working_dir, file_name)
    resume = process_resume(file_path)
    job_description = job_and_description[job]
    all_docs = {
    'resume': resume,
    'job_description': job_description
      }
    vectorstore = create_vectorstore(all_docs)
    # Create a retriever
    retriever = vectorstore.as_retriever()
    # Create a RetrievalQA chain
    qa = RetrievalQA.from_llm(llm, retriever = retriever)
    # Query the chain
    query = "Given the content of the resume and the job description how well fit is the applicant for the given position? Give a score on 10 based on much the applicant is a good fit with 10 being highest and 0 being lowest, after which give your reasons for the score."
    answer  = qa.invoke({"query":query})
    st.text_area("Result of query", answer["result"],height= 600, label_visibility="hidden")
