import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

def load_knowledge_base(file_path: str):
    # Load and split knowledge data
    loader = TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed using sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    return vectordb


def build_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2,
        groq_api_key=""
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                   -You are a helpful AI assistant for a travel platform called XploreMate. 
                   -You answer user questions specifically using the context below. 
                   -Be clear and concise. Avoid generic intros like 'Based on the provided context'. If the context includes a list of features or steps, summarize up to 2–3 key points.
                   -Stay relevant to the user's question.Do NOT mention "Additionally, you can..." or marketing fluff.
                   -If the question includes "how to" or "how can I use", give a step-by-step answer in numbered points (1, 2, etc.).

                Context:
                {context}

                Question:
                {question}

                Answer:
""".strip()
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa
