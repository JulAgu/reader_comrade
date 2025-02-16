import os
import time
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def create_chunks(src_folder: str) -> Tuple[List[int], List[int]]:
    """
    This functions takes the relative source path to
    the PDFs documents for RAG. It returns a list of chunks.

    Parameters
    ----------
    src_folder: str
        The path to the src directory
    
    Returns
    -------
    chunks: list
        The list of chunks from the documents
    """
    #TODO: At the moment this function only accepts pdfs, in the future I would like it to accept htmls and .txts and to handle them automatically.
    list_docs = os.listdir(src_folder)
    print(list_docs)
    list_docs = [src_folder + "/" + i for i in list_docs if i.endswith((".pdf"))]
    list_book = [i for i in list_docs if i.startswith((src_folder + "/" + "book_"))]
    list_review = [i for i in list_docs if i not in list_book]
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    chunks_book = []
    chunks_review = []
    for doc in list_book:
        loader = PyPDFLoader(doc)
        documents = loader.load()
        chunks_book = chunks_book + text_splitter.split_documents(documents)
    
    for review_doc in list_review:
        loader = PyPDFLoader(review_doc)
        review_docs = loader.load()
        chunks_review = chunks_review + text_splitter.split_documents(review_docs)
    
    return chunks_book, chunks_review

def embed_chunks(collection_name: str, collection: Chroma , chunks: list) -> None:
    """
    TODO: DocStrings Here
    """
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        collection.add_texts([text])  # Add text to the collection
        print(f"[{collection_name}] Chunk {i+1}/{len(chunks)} embedded.")
        time.sleep(5)

def create_databases(book: List, review: List)-> None:
    """
    TODO: DocStrings Here
    """
    # Instantiation of the OpenAI embeddings model
    embeddings = MistralAIEmbeddings()
    if len(book) > 0:
        db_name = "book_db"
        # Create a collection for the main book
        book_collection = Chroma(collection_name=db_name,
                                 persist_directory="./book_db",
                                 embedding_function=embeddings)
        embed_chunks(db_name, book_collection, book)
    else:
        print("There is not a book to embeed in the system. Please, Verify the src directory.")

    if len(review) > 0:
        db_name = "review_db"
        # Create a collection for the main book
        review_collection = Chroma(collection_name=db_name,
                                   persist_directory="./review_db",
                                   embedding_function=embeddings)
        embed_chunks(db_name, review_collection, review)
    else:
        print("There is not extra files :D.")


if __name__ == "__main__":
    chunks_book, chunks_review = create_chunks("src")
    print(len(chunks_book), len(chunks_review))
    create_databases(chunks_book, chunks_review)