import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF

try:
    from langchain_community.llms import Ollama
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    raise

def setup_conversation_chain(vector_store):
    try:
        llm = Ollama(model="qwen2:1.5b")
    except Exception as e:
        st.error(f"Error initializing Ollama: {e}")
        st.error("Ensure that the model is correctly pulled and available.")
        raise

    bot_prompt = """
    You are Safeguard, a compassionate and reliable assistant dedicated to women's safety and well-being. 
    1)Your primary role is to provide support, guidance, and information related to personal safety, emergency protocols, and resources for women in need.
    2)You are equipped to offer advice on safety tips, emergency contact information, local support services, and self-defense techniques. 
    3)You aim to empower users with knowledge and resources to help them feel safe and secure.
    4)When responding, ensure that your answers are sensitive, encouraging, and supportive. 
    5)Always prioritize the user's safety and well-being, and provide clear instructions or direct them to relevant resources if needed. 
    6)If a situation seems urgent or requires professional assistance, advise them to contact emergency services immediately.
    Your goal is to be a trusted ally in ensuring safety and providing helpful information in a reassuring and non-judgmental manner.
    """

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    def bot_conversation_chain(input_dict):
        question = input_dict["question"]
        chat_history = input_dict.get("chat_history", [])
        
        bot_question = f"{bot_prompt}\n\nHuman: {question}\nAlex:"
        
        try:
            result = conversation_chain({"question": bot_question, "chat_history": chat_history})
        except Exception as e:
            st.error(f"Error during conversation: {e}")
            return {"answer": "Sorry, I encountered an error."}

        return result

    return bot_conversation_chain

def load_conversation_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def chat(message, history, conversation_chain, history_file):
    langchain_history = [(h[0], h[1]) for h in history]
    
    try:
        response = conversation_chain({"question": message, "chat_history": langchain_history})
    except Exception as e:
        st.error(f"Error generating response: {e}")
        response = {"answer": "Sorry, I encountered an error."}

    history.append((message, response['answer']))
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    return response['answer']

def load_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

def create_faiss_index(db_path, documents_path):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")  # Example model_name
    
    try:
        document_text = load_pdf(documents_path)
        documents = [Document(page_content=document_text)]
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        raise
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
    except ImportError as e:
        st.error("FAISS is not installed. Please install it using `pip install faiss-cpu` or `pip install faiss-gpu`.")
        raise
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        raise
    
    vector_store.save_local(db_path)
    
    return vector_store

def main():
    db_path = "faiss_index"
    history_file = "conversation_history.json"
    documents_path = "Women’s Rights in India complete_compressed.pdf"  # Update this with the path to your documents
    
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")  # Example model_name
    
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        vector_store = create_faiss_index(db_path, documents_path)
    else:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    conversation_chain = setup_conversation_chain(vector_store)

    # Streamlit UI
    st.title("Women's Safety Assistant")
    st.subheader("Ask me anything about policies and practices!")
    
    chatbot = st.empty()
    msg = st.text_input("Type your question here", "")
    clear = st.button("Clear Conversation")
    
    history = load_conversation_history(history_file)

    if clear:
        if os.path.exists(history_file):
            os.remove(history_file)
        history = []
        chatbot.text("")

    if msg:
        response = chat(msg, history, conversation_chain, history_file)
        history.append((msg, response))
        chatbot.text_area("Chat History", value="\n".join([f"You: {h[0]}\nAssistant: {h[1]}" for h in history]), height=300)

if __name__ == "__main__":
    main()
