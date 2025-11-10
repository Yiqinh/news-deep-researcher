from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import sys
import json
from datetime import datetime
#from tqdm import tqdm

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
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Name or path of the Qwen LLM model to use for reasoning (e.g., Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct for CPU, Qwen/Qwen2.5-7B-Instruct for GPU)"
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


def create_query_generation_prompt(priors, article, starting_query, target):
    return f"""
You are assisting a journalist in constructing *next-step* search queries to find relevant sources
from a large corpus of embedded news articles (vector database).

Your goal: simulate what the journalist would search **before discovering the Target Source**, while
reasoning about *how* that Target Source was later used in the article.

---

## Multi-Stage Reasoning Process

### Stage 1 — Identify Article-Grounded Information Gaps
Read the article and the list of prior sources.

List **3–5 information needs** that are still unresolved.

Each gap must:
- Be grounded in the article (point to the sentence/paragraph that creates the need).
- NOT be fully covered by any of the **Prior Sources** (same topic/domain).
- Be phrased as a natural information-seeking question (e.g. “What did city officials say about the closure?”).
- Prefer gaps that a journalist would reasonably try to fill next.

Format each gap like:
- "What ... ? (motivated by: <short quote or sentence from article>)"

---

### Stage 1.5 — Align With the Target Source
You now see the **Target Source (summary only)**.

First, **generalize** the target to its purpose (e.g. “official agency statement on layoffs”, “detailed background on the policy”, “police incident report”) rather than copying its wording.

Then:
1. Decide which of the Stage 1 gaps this kind of source would actually help to fill.
2. For each selected gap, explain in 1–2 sentences how the article uses a source like this
   (e.g. “adds an official explanation”, “provides numbers”, “supplies a quote from an authority”).
3. Do **not** copy the target source title or unique wording.


---

### Stage 2 — Generate Search Queries
For **each** gap that the Target Source could fill, generate **one realistic journalist-style search query** (≤ 15 words).

**Important balance:**
- Be specific enough to plausibly surface a source like the target.
- But do **not** hard-code the exact target or outlet.
- Do not include more than 3 specificity anchors in a single query (anchors = person/organization, topic/bill/event, date/time window, document type, location etc)


You **may** use:
- Proper nouns, entities, places, and dates that appear in the **Article Context**.
- Document-type hints (e.g. “press release”, “police report”, “statement”, “hearing transcript”).

You **must not** use:
- Exact outlet names or domains that already appear in **Prior Sources**.
- Exact target source wording from the summary.


Each query must:
- Address its gap directly.
- Be semantically distinct from the others.
- Reflect an *information need*, not a known answer.

---

## Input Information
- **Starting Query:** {starting_query}
- **Article Context:** {article}
- **Prior Sources (title, domain, brief topic):** {priors}
- **Target Source (summary only):** {target}

---

## Output Format
Return **only** a single JSON object:

{{
  "information_gaps": [
    {{
      "gap": "What did city officials say about the closure?",
      "motivated_by": "City officials announced... but no statement is quoted."
    }},
    {{
      "gap": "How did regulators justify the decision?",
      "motivated_by": "The article says the decision followed a review..."
    }}
  ],
  "filled_gaps": [
    {{
      "gap": "What did city officials say about the closure?",
      "how_used_in_article": "Article uses the source to quote the city's justification and timing of the closure."
    }}
  ],
  "queries": [
    {{
      "gap": "What did city officials say about the closure?",
      "reasoning": "The target appears to be an official statement explaining the closure, so we search for a city/government statement on that event, excluding already used outlets.",
      "query": "city statement on [event/closure] explaining reasons and timing"
    }}
  ]
}}
"""

def llm_as_judge_prompt():
    return 
    
def parse_response(priors, article, starting_query, target, full_response):
    try:
        parsed_response = json.loads(full_response)
        queries_only = parsed_response.get('queries', [])
    except json.JSONDecodeError:
        queries_only = []
        parsed_response = None
        print("Warning: Could not parse full_response JSON")
    
    # Create the structured response object
    response_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        },
        "input_data": {
            "priors": priors,
            "article": article,
            "starting_query": starting_query,
            "target": target
        },
        "llm_response": {
            "full_response": full_response,
            "parsed_response": parsed_response,
            "queries_only": queries_only
        }
    }
    
    return response_data['llm_response']['queries_only']

def call_llm(prompt, model_name="Qwen/Qwen2.5-3B-Instruct"):
    print(f"Loading Qwen model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Check if CUDA is available, otherwise use CPU with appropriate dtype
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # For CPU, use float32 for better compatibility
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to("cpu")
    
    # Format prompt for Qwen using chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    response = extract_response(response, model_name)
    return response



def extract_response(response, model_name):
    """
    Extract the assistant's response from the full generated text.
    Handles Qwen model formats.
    """
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    elif "<|im_end|>" in response:
        parts = response.split("<|im_end|>")
        if len(parts) > 1:
            response = parts[-2].strip() if parts[-1].strip() == "" else parts[-1].strip()
    elif "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    return response


def get_source_names(retrieval_results):
    """Extract source names from retrieval results"""
    source_names = []
    
    """
    if retrieval_results:
        first_metadata = retrieval_results[0].get('metadata', {})
        first_source = first_metadata.get('source', {})
        print(f"[DEBUG] First result metadata keys: {list(first_metadata.keys())}")
        print(f"[DEBUG] First result source keys: {list(first_source.keys())}")
        print(f"[DEBUG] First result source: {first_source}")
        #print(f"[DEBUG] First result full metadata: {first_metadata}")
    
    """

    for doc_result in retrieval_results:
        metadata = doc_result.get('metadata', {})
        source = metadata.get('source', {})
        source_name = source.get('Name', '')  #
        source_names.append(source_name)
    
    # Print once after building the list
    print(f"[DEBUG] Retrieved source names: {source_names}")
    return source_names


def main():
    # Parse arguments
    args = parse_args()
    llm_model_name = args.llm_model_name
    
    # load combined dataset 
    combined_dataset_path = os.path.join(proj_root, 'data', 'v1_data', 'expanded_train_10000.json')    
    if not os.path.exists(combined_dataset_path):
        print(f"Combined dataset file not found at: {combined_dataset_path}")
        return
    with open(combined_dataset_path, 'r', encoding='utf-8') as f:
        combined_dataset = json.load(f)
    print(f"[DEBUG] Loaded {len(combined_dataset)} datapoints from combined dataset")

    
    # Load LLM model
    print(f"[DEBUG] Loading LLM model: {llm_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
    
    # Check if CUDA is available, otherwise use CPU with appropriate dtype
    if torch.cuda.is_available():
        print("[DEBUG] CUDA is available")
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # For CPU, use float32 for better compatibility
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to("cpu")
    print("[DEBUG] Model loaded successfully!")

    for datapoint in combined_dataset[:1]:
        
        article = datapoint['article']['article_text']
        #print(f"Article: {article}")
        starting_query = datapoint['starting_query']['model_output']
        #print(f"Starting query: {starting_query}")
        priors = datapoint['prior_sources']
        #print(f"Priors: {priors}")
        target = datapoint['target_source']
        #print(f"Target: {target}")
        target_source_name = datapoint['target_source']['Name']

        print("[DEBUG] sending prompt to qwen")
        query_generation_prompt = create_query_generation_prompt(priors, article, starting_query, target)
        messages = [{"role": "user", "content": query_generation_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = extract_response(response, llm_model_name)

        # Save response to file
        response_file_path = os.path.join(proj_root, 'results', 'qwen_test_response.jsonl')
        os.makedirs(os.path.dirname(response_file_path), exist_ok=True)
        
        # Save as text file
        with open(response_file_path, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print("\n Qwen Response")
        print(response)
        print(f"\nResponse saved to: {response_file_path}")


        queries_only = parse_response(priors, article, starting_query, target, response)

        # Initialize searcher
        news_searcher = Searcher(index_name=args.index_name, model_name=args.model_name)

        # Call retriever for each query
        for query_dict in queries_only:
            query = query_dict['query']
            print(f"[DEBUG] Searching with query: {query}")
            
            # Retrieve
            document_list = news_searcher.search(query=query, k=args.k)
            retrieval_result = []
            for doc in document_list:
                one_doc = {'page_content': doc.page_content, 'metadata': doc.metadata}
                retrieval_result.append(one_doc)
            
            # Get and print source names
            source_names = get_source_names(retrieval_result)

            #check if target source is in the retrieval result
            for source_name in source_names:
                if source_name == target_source_name:
                    print(f"[DEBUG] Target source found in retrieval result")
                    break
                else:
                    print(f"[DEBUG] Target source not found in retrieval result")
                    break



    


if __name__ == "__main__":
    main()

