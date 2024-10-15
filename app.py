import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModel


openai.api_key = st.secrets["OPENAI_API_KEY"]

# Specify the Hugging Face model for embeddings (e.g., sentence-transformers model)
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load the Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load existing vector store with Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Define the path to your FAISS vector store
faiss_store_path = "faiss_vector_store"

# Load the existing FAISS vector store
vectorstoredb = FAISS.load_local(faiss_store_path, embeddings, allow_dangerous_deserialization=True)

# Use the FAISS vector store as a retriever
retriever = vectorstoredb.as_retriever()


llm = ChatOpenAI(model="gpt-4o-mini")


system_prompt = (
    "You are a chatbot designed to assist users with information related to the Pilihan Raya (elections) in Malaysia." 
    "Your primary task is to provide responses based on the official data from the Suruhanjaya Pilihan Raya Malaysia (Election Commission of Malaysia) website." 
    "Always rely on the given context information to respond to user queries first." 
    "If sufficient information is available in the provided context, only use that to answer the question." 
    "If there is insufficient data in the context, you may use your general language understanding to fill in gaps while ensuring your response remains accurate, concise, and reliable." 
    "Determine the language of the user's query." 
    "If the query is in English, respond in English." 
    "If it is in Bahasa Melayu, respond in Bahasa Melayu." 
    "Begin each response with a friendly greeting, such as 'Hello!' or 'Hi there!' for English queries, and 'Halo!' or 'Hai!' for Bahasa Melayu queries to engage the user." 
    "For questions about elections, use the context to explain what pilihan raya is, including its significance in Malaysia's democratic process, in the appropriate language." 
    "For questions about specific election dates, provide relevant dates and any associated events, such as nomination days or polling days, based on the context and the user's language." 
    "If a user asks about the voting process, outline the steps involved, such as registration, where to vote, and what to bring on polling day, using the provided context." 
    "When asked about candidates, explain how to find reliable information about candidates contesting in the elections using data from the context." 
    "If the query is about voter eligibility, use the context to explain the criteria to vote, including age, citizenship, and registration requirements in the appropriate language." 
    "If any information is not available in the context, kindly inform the user that you might not have the answer and suggest visiting the official SPR website for more details in their preferred language." 
    "End responses with an invitation for further questions, like 'Is there anything else I can help you with?' for English queries and 'Ada lagi yang boleh saya bantu?' for Bahasa Melayu queries."
    "\n\n"
    "{context}"
)


# Define the prompt template for OpenAI models using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)




# Streamlit interface
st.title("SPR Malaysia Pilihan Raya Chatbot")

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial greeting from the assistant
    initial_message = "Hello! I'm here to assist you with any questions you have about pilihan raya in Malaysia. Whether you need information about the voting process, candidates, or anything else, feel free to ask!"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display initial assistant message with logo
if st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[0]["content"])

# Display chat messages in collapsible sections (accordion-style)
for idx, message in enumerate(st.session_state.messages[1:], start=1):
    if message["role"] == "user":
        with st.expander(f"Question {idx // 2 + 1}: {message['content'][:50]}...", expanded=False):
            st.write(f"**User:** {message['content']}")
            if idx + 1 < len(st.session_state.messages):
                st.write(f"**Assistant:** {st.session_state.messages[idx + 1]['content']}")

# Accept user input
if prompt := st.chat_input("Ask about pilihan raya in Malaysia..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = rag_chain.invoke({"input": prompt})
        response = result.get('answer', "I'm not sure.")
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})