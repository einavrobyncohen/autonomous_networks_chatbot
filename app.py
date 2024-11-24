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
    # Stylish Header
    st.markdown("""
    <div style="background-color: #2E86C1; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white;">The Autonomous Network Chatbot 🌟</h1>
        <p style="color: white; font-size: large;">Your one-stop bot for unraveling the secrets of autonomous networks!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with Overview
    st.sidebar.title("🤔 Umm.. Autonomous Networks?")
    st.sidebar.info("""
    Autonomous networks are smart, self-managing systems designed to configure, optimize, and heal themselves with minimal human intervention.   
    """)

    # Introduction and Example Queries
    st.markdown("""
    <div style="margin-top: 20px; padding: 10px; border-radius: 10px; background-color: #F7F9F9;">
        <h3>Examples of Things You Can Ask Me:</h3>
        <ul style="font-size: large; line-height: 1.8;">
            <li>💡 <b>What are autonomous networks?</b></li>
            <li>🔍 <b>What are the benefits of intent-based networking?</b></li>
            <li>🔐 <b>How do autonomous networks ensure security?</b></li>
            <li>⚙️ <b>What are the challenges in implementing zero-touch automation?</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer Section
    st.info("""
    **Disclaimer:**  
    While I strive for accuracy, I might make mistakes—I'm just a fancy AI bot, after all! 🤖  
    Always double-check critical information before using it for important decisions.
    """)

    # Initialize the bot
    if "bot" not in st.session_state:
        with st.spinner("Initializing the Whisperer..."):
            st.session_state.bot = AutonomousNetworkBot()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History with Stylish Messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #E8F6F3; padding: 10px; border-left: 5px solid #48C9B0; border-radius: 5px; margin-bottom: 10px;">
                🧑‍💻 **User:**  
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div style="background-color: #F9F9F9; padding: 10px; border-left: 5px solid #2E86C1; border-radius: 5px; margin-bottom: 10px;">
                🤖 **Bot:**  
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Ask me anything about autonomous networks!"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
        <div style="background-color: #E8F6F3; padding: 10px; border-left: 5px solid #48C9B0; border-radius: 5px; margin-bottom: 10px;">
            🧑‍💻 **User:**  
            {prompt}
        </div>
        """, unsafe_allow_html=True)

        # Get and display bot response
        with st.spinner("Let me think..."):
            chat_history = [(m["content"], m["role"]) for m in st.session_state.messages[:-1]]
            response = st.session_state.bot.get_response(prompt, chat_history)

            st.markdown(f"""
            <div style="background-color: #F9F9F9; padding: 10px; border-left: 5px solid #2E86C1; border-radius: 5px; margin-bottom: 10px;">
                🤖 **Bot:**  
                {response["answer"]}
            </div>
            """, unsafe_allow_html=True)

        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})


if __name__ == "__main__":
    main()
