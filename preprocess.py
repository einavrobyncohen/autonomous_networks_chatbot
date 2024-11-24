# preprocess.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os

def preprocess_documents():
    print("Starting preprocessing...")
    
    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load documents
    print("Loading documents...")
    loader = DirectoryLoader("processed_data", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Create and persist vector store
    print("Creating vector store...")
    if os.path.exists("db"):
        import shutil
        shutil.rmtree("db")
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="db"
    )
    vectorstore.persist()
    
    print("Preprocessing complete! Vector store saved to 'db' directory")

if __name__ == "__main__":
    preprocess_documents()