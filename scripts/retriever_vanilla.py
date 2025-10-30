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
        default="Qwen/Qwen3-Embedding-8B",
        help="Name or path of the embedding model to use"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default='../faiss_index_news_sources',
        help="path to the faiss index"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default='./data/query_generation_results_2.jsonl',
        help="path to file with starting queries"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='retriever_output.json',
        help="output path"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top results to retrieve"
    )
    return parser.parse_args()

def main(model_name, index_name, query_file, output_path, k):

    # load queries
    print(f"Reading data from {query_file}...")

    datapoints = []

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            point = json.loads(line)
            datapoints.append(point)
        print(f"loaded {len(datapoints)} points...")

    # load index
    news_searcher = Searcher(index_name=index_name, model_name=model_name)

    for point in tqdm(datapoints, desc="retrieving documents..."):
        for query_dict in point['llm_response']['queries_only']:
            query = query_dict['query']

             # retrieval
            document_list = news_searcher.search(query=query, k=k)
            retrieval_result = []
            for doc in document_list:
                one_doc = {'page_content': doc.page_content, 'metadata': doc.metadata}
                retrieval_result.append(one_doc)

            query_dict['retrieval'] = retrieval_result

    with open(output_path, "w", encoding="utf-8") as f:
        print("saved retrieval results: ", output_path)
        json.dump(datapoints, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, args.index_name, args.query_file, args.output_path, args.k)


