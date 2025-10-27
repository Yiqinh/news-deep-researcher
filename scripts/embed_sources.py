import argparse
import json
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def main(model, data_file, index_name):
    # load model
    embeddings = HuggingFaceEmbeddings(model_name=model)
    # load retrieval documents
    documents = load_documents(data_file)
    # make vectorstore
    print(f"Embedding Model: {model}")
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("saving index to: ", index_name)
    vectorstore.save_local(index_name)

def load_documents(data_file):
    # load source documents from json file
    print(f"Loading Dataset:  {data_file}")
    with open(data_file, "r") as f:
        articles = json.load(f)
    
    document_list = []
    for news_article in articles:
        for source in news_article['sources']:
            # append all source information to embedding
            embed_string = ""
            for k, v in source.items():
                embed_string += v
                embed_string += '\n'
            # make document
            source_metadata = news_article['sources_metadata']
            source_metadata['source'] = source
            source_metadata['starting_query'] = news_article['starting_query']
            source_metadata['article'] = news_article['article']
            source_document = Document(
                page_content=embed_string,
                metadata=source_metadata
            )
            document_list.append(source_document)

    return document_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Name or path of the embedding model to use"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default='./data/v1_data/full_news_dataset_v1.json',
        help="Path to the data file to process"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default='faiss_index',
        help="name of faiss output dir"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Embedding Model: ", args.model)
    main(model=args.model, data_file=args.data_file, index_name=args.index_name)