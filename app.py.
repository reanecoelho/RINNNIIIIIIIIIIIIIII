import streamlit as st
import os
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Doc Q&A Bot", layout="centered")
st.title("AI-Powered Document Q&A Bot 🤖")

# --- CORE FUNCTIONS ---

@st.cache_resource
def load_llm_and_embeddings():
    """Load the Language Model and Embedding Model once."""
    # Note: This requires the HUGGINGFACEHUB_API_TOKEN to be set in Streamlit's secrets
    llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                       task="text2text-generation",
                       model_kwargs={"temperature": 0.6, "max_length": 1024})
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return llm, embeddings

@st.cache_data
def create_vector_store_from_ppt(uploaded_file):
    """Create a vector store from the uploaded PowerPoint file."""
    if uploaded_file is not None:
        # To read the file, we need to save it temporarily
        temp_file_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = UnstructuredPowerPointLoader(temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        
        _, embeddings = load_llm_and_embeddings()
        db = FAISS.from_documents(texts, embeddings)
        return db
    return None

# --- UI AND LOGIC ---

# Load models (this will only run once)
llm, embeddings = load_llm_and_embeddings()

# File uploader
st.header("1. Upload Your PowerPoint Document")
uploaded_file = st.file_uploader("Upload a .pptx file", type=["pptx"])

# Create vector store if a file is uploaded
if uploaded_file:
    with st.spinner("Processing your document... This may take a moment."):
        vector_store = create_vector_store_from_ppt(uploaded_file)
    st.success("Document processed successfully! You can now ask questions.")
    
    # Store the vector_store in session state to persist it
    st.session_state.vector_store = vector_store

# Question input
st.header("2. Ask a Question")
if 'vector_store' in st.session_state:
    query = st.text_input("Enter your question about the document:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching for the answer..."):
                # Retrieve the vector store from session state
                db = st.session_state.vector_store
                retriever = db.as_retriever(search_kwargs={"k": 3})
                
                # Create the Q&A chain
                qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                      chain_type="stuff",
                                                      retriever=retriever,
                                                      return_source_documents=False)
                
                # Get the result
                result = qa_chain({"query": query})
                
                # Display the answer
                st.subheader("Answer:")
                st.write(result["result"])
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a document to begin.")
