import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
import chromadb
from chromadb.config import Settings

class AutonomousNetworkBot:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Initialize LLM
        self.llm = Ollama(model="mistral")
        
        # Initialize ChromaDB with specific settings
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db"
        ))
        
        # Load existing vector store
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings,
            persist_directory="db"
        )
        
        # Setup retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

    def get_response(self, query, chat_history):
        return self._get_cached_response(query, chat_history, self.chain)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def _get_cached_response(_query, _chat_history, _chain):
        prompt = f"""
        You are an expert on autonomous networks. Be concise and to the point. 
        If the question is not about autonomous networks, respond with a single 
        sentence declining to answer.

        Question: {_query}
        """
        
        result = _chain({"question": prompt, "chat_history": _chat_history})
        return result["answer"]

def main():
    # Page config
    st.set_page_config(
        page_title="Autonomous Networks Expert",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Autonomous Networks Expert 🤖")
    st.write("Ask me anything about autonomous networks!")
    
    # Initialize bot
    if "bot" not in st.session_state:
        with st.spinner("Initializing bot..."):
            st.session_state.bot = AutonomousNetworkBot()
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about autonomous networks?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = [(m["content"], m["role"]) for m in st.session_state.messages[:-1]]
                response = st.session_state.bot.get_response(prompt, tuple(chat_history))
                st.markdown(response)
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()