import os
from config import config, VECTOR_STORE_DIR
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=config.base_config.OPENAI_API_KEY)


def pdf_loader(pdf) -> list[str]:
    """
    read pdf and make small chunks for pdf text
    """
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def save_vector_store(text_chunks: list[str], save: bool = True,
                      output_path: str = os.path.join(VECTOR_STORE_DIR,
                                                      "vectorstore")):
    store = FAISS.from_texts(text_chunks, embeddings)
    if save:
        store.save_local(output_path)
    return store


def load_vector_store(
        input_path: str = os.path.join(VECTOR_STORE_DIR, "vectorstore")):
    if not os.path.exists(input_path):
        return None
    store = FAISS.load_local(input_path, embeddings,
                             allow_dangerous_deserialization=True)
    return store
