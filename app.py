import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import os
from typing import List, Tuple, Dict

# Azure OpenAI configurations
if ('AZURE_OPENAI_API_KEY' not in st.secrets and 'AZURE_OPENAI_API_KEY' not in os.environ) or \
   ('AZURE_OPENAI_ENDPOINT' not in st.secrets and 'AZURE_OPENAI_ENDPOINT' not in os.environ) or \
   ('AZURE_OPENAI_DEPLOYMENT_NAME' not in st.secrets and 'AZURE_OPENAI_DEPLOYMENT_NAME' not in os.environ):
    st.error('Azure OpenAI credentials not found. Please add API key, endpoint, and deployment name to your Streamlit secrets or environment variables.')
    st.stop()

azure_api_key = st.secrets.get('AZURE_OPENAI_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
azure_endpoint = st.secrets.get('AZURE_OPENAI_ENDPOINT') or os.environ.get('AZURE_OPENAI_ENDPOINT')
azure_deployment = st.secrets.get('AZURE_OPENAI_DEPLOYMENT_NAME') or os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an expert on autonomous networks with deep technical knowledge. Use the following context and chat history to provide a detailed, specific answer to the question. Focus on being accurate and informative while avoiding generic responses.

Context: {context}

Chat History: {chat_history}

Question: {question}

Important Guidelines:
1. If the question is not about autonomous networks, politely decline to answer
2. Use specific examples and technical details from the provided context
3. If different sources provide conflicting information, acknowledge it
4. If the context doesn't contain enough information to fully answer the question, acknowledge the limitations
5. Maintain a professional, technical tone while being clear and concise

Answer:"""
)

class AutonomousNetworkBot:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            deployment_name=azure_deployment,
            openai_api_key=azure_api_key,
            openai_api_version="2023-05-15",
            azure_endpoint=azure_endpoint,
            temperature=0.3  # Reduced temperature for more focused responses
        )
        
        # Initialize ChromaDB with specific settings
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db"
        ))
        
        # Load vector store
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings,
            persist_directory="db"
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Setup retriever with search parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance
            search_kwargs={
                "k": 4,  # Retrieve more documents
                "fetch_k": 8,  # Consider a larger initial set
                "lambda_mult": 0.7  # Diversity factor
            }
        )
        
        # Initialize chain with custom prompt
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            return_source_documents=True,
            verbose=True
        )

    def get_response(self, query: str, chat_history: List[Tuple[str, str]]) -> Dict:
        """Get response with enhanced context handling and filtering."""
        # Check if query is about autonomous networks
        if not self._is_relevant_query(query):
            return {
                "answer": "I can only answer questions about autonomous networks. Please ask a question related to autonomous networks.",
                "sources": []
            }

        # Get response from chain
        result = self.chain({"question": query, "chat_history": chat_history})
        
        # Process and validate response
        processed_response = self._process_response(result["answer"], query)
        
        return {
            "answer": processed_response,
            "sources": result.get("source_documents", [])
        }

    def _is_relevant_query(self, query: str) -> bool:
        """Check if query is related to autonomous networks."""
        relevant_terms = [
            "autonomous", "network", "ai", "automation", "self-managing",
            "self-healing", "intent-based", "zero-touch", "machine learning",
            "orchestration", "sdn", "nfv", "telemetry", "analytics"
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in relevant_terms)

    def _process_response(self, response: str, query: str) -> str:
        """Process and validate response."""
        # Check for generic responses
        generic_patterns = [
            "autonomous networks are self-managing networks",
            "autonomous networks utilize artificial intelligence",
            "autonomous networks are networks that"
        ]
        
        if any(pattern in response.lower() for pattern in generic_patterns):
            # If response is too generic, try to make it more specific
            specific_terms = {
                "advantages": "specific benefits include",
                "benefits": "key advantages are",
                "components": "main components include",
                "architecture": "the architecture consists of",
                "challenges": "key challenges include",
                "implementation": "implementation involves",
                "features": "important features include"
            }
            
            for term, prefix in specific_terms.items():
                if term in query.lower():
                    return f"{prefix} {response}"
        
        return response

def main():
    st.set_page_config(
        page_title="The Autonomous Network Whisperer 🕵️‍♂️",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("The Autonomous Network Whisperer 🌟")
    st.write("""
    Welcome to **The Autonomous Network Whisperer**! 🤓  
    I'm here to answer your most curious and complex questions about **autonomous networks**,  
    whether it's about their architecture, challenges, or benefits.
    
    I might not have all the answers, but I’ll do my best to **wow you with my AI-powered knowledge!** 🚀  
    If you’re curious about how I work or want to dive into the code and data that powers me,  
    check out my developer's [GitHub repository](https://github.com/einavrobyncohen/autonomous_networks_chatbot).  
    """)
    
    st.info("""
    **Disclaimer:**  
    While I strive for accuracy, I might make mistakes—I'm just a fancy AI bot, after all! 🤖  
    Always double-check critical information before using it for important decisions.
    """)

    # Initialize bot
    if "bot" not in st.session_state:
        with st.spinner("Initializing the Whisperer..."):
            st.session_state.bot = AutonomousNetworkBot()
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What autonomous network secrets shall we uncover today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Let me think..."):
                chat_history = [(m["content"], m["role"]) for m in st.session_state.messages[:-1]]
                response = st.session_state.bot.get_response(prompt, chat_history)
                st.markdown(response["answer"])
                
                if response["sources"]:
                     st.write("Sources:")
                     for source in response["sources"]:
                         st.write(f"- {source.metadata.get('source', 'Unknown source')}")
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()