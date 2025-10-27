import json
import itertools
from typing import List, Dict, Any, Iterator
from tqdm import tqdm
import random as random

def generate_source_combinations(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_datapoints = []
    sources = article.get('sources', [])
    n = len(sources)
    
    # Pre-compute the base article (without sources, target_source, prior_sources)
    base_article = {k: v for k, v in article.items() if k not in ['target_source', 'prior_sources']}
    
    # For each source as the target
    for target_idx in range(n):
        target_source = sources[target_idx]
        remaining_sources = [s for i, s in enumerate(sources) if i != target_idx]
        
        if n == 1:
            new_article = base_article.copy()  # shallow copy of base
            new_article['target_source'] = target_source
            new_article['prior_sources'] = []
            new_article['sources'] = sources
            new_datapoints.append(new_article)
        else:
            # For each possible size k (from 0 to n-1)
            for k in range(0, n):
                # Generate all combinations of size k from remaining sources
                for combo in itertools.combinations(remaining_sources, k):
                    new_article = base_article.copy()
                    new_article['target_source'] = target_source
                    new_article['prior_sources'] = list(combo)
                    new_article['sources'] = sources 
                    new_datapoints.append(new_article)
    
    return new_datapoints

def expand_dataset_batch(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standard expansion - loads all into memory
    """
    all_articles = []    
    for article in tqdm(data, desc="Processing articles"):
        combinations = generate_source_combinations(article)
        all_articles.extend(combinations)
    return all_articles


def process_json_file(input_file: str, output_file: str, method: str = "batch", num_points=10000):

    print(f"Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} articles")
    
    combinations = expand_dataset_batch(data)
    random.shuffle(combinations)
    combinations = combinations[:num_points]

    with open(output_file, 'w') as f:
        json.dump(combinations, f, indent=2)
    print(f"Number of datapoints (Combinations): {len(combinations)}")

if __name__ == "__main__":
    input_file = "/Users/yiqinhuang/NLP/news-deep-researcher/notebooks/news_data_train.json"
    output_file = "expanded_train_10000.json"
    
    process_json_file(input_file, output_file)
    