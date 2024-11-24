from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import UnstructuredFileLoader
import os
import re
from typing import List, Dict

def extract_metadata_from_content(text: str) -> Dict[str, str]:
    """Extract metadata from document content."""
    metadata = {}
    
    # Extract section title if present
    section_match = re.search(r'^#\s*(.+)$', text, re.MULTILINE)
    if section_match:
        metadata['section'] = section_match.group(1)
    
    # Identify document type based on content patterns
    if re.search(r'architecture|design|structure', text, re.I):
        metadata['doc_type'] = 'technical'
    elif re.search(r'implement|deploy|install', text, re.I):
        metadata['doc_type'] = 'implementation'
    elif re.search(r'overview|introduction|background', text, re.I):
        metadata['doc_type'] = 'conceptual'
    
    return metadata

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
    loader = DirectoryLoader(
        "processed_data",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    documents = loader.load()
    
    # Split documents with better chunking strategy
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for more precise retrieval
        chunk_overlap=200,  # Larger overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False
    )
    texts = text_splitter.split_documents(documents)
    
    # Enhance documents with metadata
    print("Enhancing documents with metadata...")
    for doc in texts:
        doc.metadata.update(extract_metadata_from_content(doc.page_content))
    
    # Create and persist vector store
    print("Creating vector store...")
    if os.path.exists("db"):
        import shutil
        shutil.rmtree("db")
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="db",
        collection_metadata={"hnsw:space": "cosine"}  # Optimize for semantic search
    )
    vectorstore.persist()
    
    print("Preprocessing complete! Vector store saved to 'db' directory")

if __name__ == "__main__":
    preprocess_documents()