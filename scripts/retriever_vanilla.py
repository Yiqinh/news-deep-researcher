import argparse
import os
import sys
import json
from tqdm import tqdm

# this file
here = os.path.dirname(os.path.abspath(__file__))
# proj root
proj_root = os.path.dirname(here)
# add proj root to path
sys.path.append(proj_root)

from src.searcher import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Name or path of the embedding model to use"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default='./faiss_index_qwen4b',
        help="path to the faiss index"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default='./data/combined_queries.jsonl',
        help="path to file with starting queries"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='retriever_output.json',
        help="output path"
    )
    return parser.parse_args()

def main(model_name, index_name, query_file, output_path):

    # load queries
    print(f"Reading data from {query_file}...")

    id_to_queries = {}
    query_keys = set()

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            custom_id = data.get('custom_id')
            custom_id = custom_id[-5:] # get last 5 digits
            generation_content = data.get('model_output')
                        
            if custom_id and generation_content:
                # map custom_id to content
                id_to_queries[custom_id] = generation_content
                query_keys.add(custom_id)
            else:
                # skip if no custom id or no gen content
                continue
    
    # load index
    news_searcher = Searcher(index_name=index_name, model_name=model_name)

    # store source custom keys
    datastore_keys = set()
    
    # loop through custom keys in datastore
    for doc_id, document in news_searcher.vectorstore.docstore._dict.items():
        metadata = document.metadata
        custom_id_string = metadata['custom_id']
        custom_id = custom_id_string[-5:]
        datastore_keys.add(custom_id)
    
    id_to_retrieval = {}
    for id, query in tqdm(list(id_to_queries.items()), desc="retrieving documents..."):
        if id not in datastore_keys:
            # skip ids not in the datastore
            continue

        document_list = news_searcher.search(query=query, k=10)
        retrieval_result = []

        for doc in document_list:
            one_doc = {'page_content': doc.page_content, 'metadata': doc.metadata}
            retrieval_result.append(one_doc)

        id_to_retrieval[id] = retrieval_result

    with open(output_path, "w", encoding="utf-8") as f:
        print("saved retrieval results: ", output_path)
        json.dump(id_to_retrieval, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, args.index_name, args.query_file, args.output_path)


