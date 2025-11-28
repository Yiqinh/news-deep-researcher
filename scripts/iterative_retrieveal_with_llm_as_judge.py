from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import sys
import json
import re
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
        default='../results/query_and_filter_sft_data_test.json',
        help="output path"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=20,
        help="Maximum number of query attempts (initial + retry queries)"
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
- **Balance specificity with flexibility**: Avoid over-constraining queries with too many specific elements (person, organization, topic, date, location, document etc.). Aim for a natural journalist-style query that would realistically surface the target, not a hyper-specific description that essentially describes the exact source.

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

def create_query_generation_retry_prompt(priors, article, starting_query, target, queries_tried, query_to_sources):
    """
    Create a gap-based retry prompt that learns from failed queries.
    """
    queries_tried_str = "\n".join([f"- '{q}'" for q in queries_tried])

    print (f"queries_tried: {queries_tried}")
    
    retrieved_sources_str = ""
    for i, query in enumerate(queries_tried, 1):
        retrieved_sources = query_to_sources.get(query, [])
        sources_json = json.dumps(retrieved_sources, indent=2, ensure_ascii=False)
        retrieved_sources_str += f"\n\nQuery {i}: '{query}'\nRetrieved sources (full dictionaries):\n{sources_json}"
    
    return f"""
You are assisting a journalist in constructing *next-step* search queries to find a specific target source
from a large corpus of embedded news articles (vector database).

**CRITICAL: Previous queries did NOT successfully retrieve the Target Source.**

---

## Previous Attempts (Failed)

The following queries were tried but did not retrieve the Target Source:

{queries_tried_str}

**What was retrieved instead (full source dictionaries):**
{retrieved_sources_str}

**Key insight:** The queries above retrieved sources that are NOT the target. Use this information to generate queries that are different and more likely to succeed. The new queries must differ significantly from all previously failed queries.
They should explore alternative angles, stakeholders, mechanisms, document types, etc — not merely rephrase the same concept. If a new query is only a rewording of an old one, discard it and generate a more distinct alternative.

---

## Multi-Stage Reasoning Process (Revised)

### Stage 1 — Re-identify Information Gaps (Learning from Failures)

Read the article and the list of prior sources. **Importantly, consider what gaps the failed queries were trying to address and why they didn't work.**

List **3–5 information needs** that are still unresolved. **Focus on gaps that:**
- Were NOT successfully addressed by the failed queries
- Require a different approach than what was tried
- Might need a different angle, specificity level, or keyword choice

Each gap must:
- Be grounded in the article (point to the sentence/paragraph that creates the need).
- NOT be fully covered by any of the **Prior Sources** (same topic/domain).
- Be phrased as a natural information-seeking question.
- **Be different from the gaps that the failed queries addressed.**

Format each gap like:
- "What ... ? (motivated by: <short quote or sentence from article>)"

---

### Stage 1.5 — Align With Target Source (Learning What to Avoid)

You now see the **Target Source (summary only)**.

First, **generalize** the target to its purpose (e.g. "official agency statement on layoffs", "detailed background on the policy", "police incident report").

**Then analyze why previous queries failed:**
1. What kind of sources did the failed queries retrieve? (e.g., news articles, different outlets, different document types)
2. How do those retrieved sources differ from the Target Source?
3. What should the new queries do differently to avoid retrieving the same wrong sources?

Then:
1. Decide which of the Stage 1 gaps this kind of source would actually help to fill.
2. For each selected gap, explain in 1–2 sentences how the article uses a source like this.
3. **Explain how your approach differs from the failed queries.**

---

### Stage 2 — Generate NEW Search Queries (Learning from Mistakes)

For **each** gap that the Target Source could fill, generate **one realistic journalist-style search query** (≤ 15 words) that:

**MUST be different from failed queries:**
- Use different keywords or phrasing
- Try a different angle or specificity level
- Focus on different aspects of the gap
- Avoid approaches that retrieved wrong sources

**Still maintain constraints:**
- Be specific enough to plausibly surface a source like the target.
- Do **not** hard-code the exact target or outlet.
- **Balance specificity with flexibility**: Avoid over-constraining queries with too many specific elements (person, organization, topic, date, location , document type etc). Aim for a natural journalist-style query that would realistically surface the target, not a hyper-specific description that essentially describes the exact source.
**You may use:**
- Proper nouns, entities, places, and dates from the **Article Context**.
- Document-type hints (e.g., "press release", "police report", "statement").

**You must not use:**
- Exact outlet names or domains from **Prior Sources**.
- Exact target source wording from the summary.
- **Similar phrasing or keywords to the failed queries above.**

**IMPORTANT** DO NOT REPEAT QUERIES FROM THE FAILED ONES.

---

## Input Information

- **Starting Query:** {starting_query}
- **Article Context:** {article}
- **Prior Sources (title, domain, brief topic):** {priors}
- **Target Source (summary only):** {target}
- **Failed Queries:** {queries_tried_str}

---

## Output Format

Return **only** a single JSON object:

{{
  "analysis": {{
    "why_previous_failed": "Brief analysis of why the previous queries didn't retrieve the target",
    "what_to_avoid": "What kind of sources/approaches to avoid based on what was retrieved"
  }},
  "information_gaps": [
    {{
      "gap": "What did city officials say about the closure?",
      "motivated_by": "City officials announced... but no statement is quoted.",
      "how_different": "This gap focuses on X, whereas failed queries focused on Y"
    }}
  ],
  "filled_gaps": [
    {{
      "gap": "What did city officials say about the closure?",
      "how_used_in_article": "Article uses the source to quote the city's justification and timing of the closure.",
      "why_this_will_work": "This approach differs from failed queries by..."
    }}
  ],
  "queries": [
    {{
      "gap": "What did city officials say about the closure?",
      "reasoning": "The target appears to be an official statement. Previous queries retrieved news articles, so this query focuses on official statements rather than news coverage.",
      "how_different": "Uses 'official statement' instead of 'news about' to avoid retrieving news articles",
      "query": "official statement city officials closure reasons"
    }}
  ]
}}
"""

def llm_as_judge_prompt():
    return 
    
def parse_response(priors, article, starting_query, target, full_response):
    parsed_response = extract_json_from_response(full_response)
    
    if parsed_response is None:
        queries_only = []
        parsed_response = None
        print("Warning: Could not parse full_response JSON")
    else:
        queries_only = parsed_response.get('queries', [])
    
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
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
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


def extract_json_from_response(response_text):
    """
    Extract JSON from LLM response, handling markdown code blocks and extra text.
    
    Args:
        response_text: The raw response text from the LLM
    
    Returns:
        Parsed JSON object, or None if extraction fails
    """
    if not response_text:
        return None
    
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    # First, find the code block markers
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        # Try to find the JSON object within the code block
        brace_start = json_str.find('{')
        if brace_start != -1:
            brace_count = 0
            brace_end = -1
            for i in range(brace_start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break
            
            if brace_end != -1:
                json_str = json_str[brace_start:brace_end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # Try to find JSON object in the text (look for { ... })
    # Find the first { and try to match it with the closing }
    brace_start = response_text.find('{')
    if brace_start != -1:
        # Try to find the matching closing brace
        brace_count = 0
        brace_end = -1
        for i in range(brace_start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i
                    break
        
        if brace_end != -1:
            json_str = response_text[brace_start:brace_end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Try to parse the entire response as JSON (in case it's already clean JSON)
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # If all else fails, try to fix common JSON issues
    # Remove trailing commas before closing braces/brackets
    fixed_text = re.sub(r',\s*}', '}', response_text)
    fixed_text = re.sub(r',\s*]', ']', fixed_text)
    
    # Try to extract JSON again from fixed text
    brace_start = fixed_text.find('{')
    if brace_start != -1:
        brace_count = 0
        brace_end = -1
        for i in range(brace_start, len(fixed_text)):
            if fixed_text[i] == '{':
                brace_count += 1
            elif fixed_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i
                    break
        
        if brace_end != -1:
            json_str = fixed_text[brace_start:brace_end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    return None


def get_retrieved_sources(retrieval_results):
    """Extract full source dictionaries from retrieval results"""
    retrieved_sources = []
    
    if retrieval_results:
        first_metadata = retrieval_results[0].get('metadata', {})
        first_source = first_metadata.get('source', {})
        #print(f"[DEBUG] First result metadata keys: {list(first_metadata.keys())}")
        #print(f"[DEBUG] First result source keys: {list(first_source.keys())}")
        #print(f"[DEBUG] First result source: {first_source}")

    for doc_result in retrieval_results:
        metadata = doc_result.get('metadata', {})
        source = metadata.get('source', {})
        retrieved_sources.append(source)
    
    # Print source names for debugging
    source_names = [s.get('Name', '') for s in retrieved_sources]
    #print(f"[DEBUG] Retrieved source names: {source_names}")
    
    return retrieved_sources


def get_all_retrieved_sources(queries_tried, all_retrieval_results):
    """
    Get all full source dictionaries retrieved from failed queries.
    """
    query_to_sources = {}
    for query, retrieval_result in zip(queries_tried, all_retrieval_results):
        retrieved_sources = get_retrieved_sources(retrieval_result)
        query_to_sources[query] = retrieved_sources
    
    return query_to_sources


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

    # Collect enhanced datapoints
    enhanced_dataset = []
    # Collect simplified datapoints (just the key fields)
    simplified_dataset = []

    news_searcher = Searcher(index_name=args.index_name, model_name=args.model_name)

    for idx, datapoint in enumerate(combined_dataset[10:25], 1):
        print(f"\n{'='*60}")
        print(f"[DEBUG] Processing datapoint {idx}/10")
        print(f"{'='*60}\n")
        
        try:
            article = datapoint['article']['article_text']
            #print(f"Article: {article}")
            # Handle different starting_query structures
            if isinstance(datapoint.get('starting_query'), dict):
                starting_query = datapoint['starting_query'].get('model_output', datapoint['starting_query'].get('query', ''))
            else:
                starting_query = str(datapoint.get('starting_query', ''))
            #print(f"Starting query: {starting_query}")
            priors = datapoint.get('prior_sources', [])
            #print(f"Priors: {priors}")
            target = datapoint['target_source']
            #print(f"Target: {target}")

            print("[DEBUG] sending prompt to qwen")
            query_generation_prompt = create_query_generation_prompt(priors, article, starting_query, target)
            messages = [{"role": "user", "content": query_generation_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)

            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            response = extract_response(response, llm_model_name)

            # Save response to file
            response_file_path = os.path.join(proj_root, 'results', 'qwen_test_10_2.jsonl')
            os.makedirs(os.path.dirname(response_file_path), exist_ok=True)
            
            # Save as text file (append mode to save all responses)
            with open(response_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Datapoint {idx} ---\n")
                f.write(response)
                f.write("\n")
            
            print("\n Qwen Response (first 500 chars):")
            print(response[:500] + "..." if len(response) > 500 else response)
            print(f"\nResponse saved to: {response_file_path}")
            print(f"[DEBUG] Response length: {len(response)} characters")

            queries_only = parse_response(priors, article, starting_query, target, response)
            
            if not queries_only:
                print(f"[WARNING] No queries extracted from response. Response preview: {response[:200]}...")

            queries_tried = []
            all_retrieval_results = []  
            target_found = False
            rank = None
            total_attempts = 0
            max_attempts = args.max_attempts
            target_query = None  
            filter_set = None 

            # Call retriever for each query
            for query_dict in queries_only:
                # Check if we've reached max attempts
                if total_attempts >= max_attempts:
                    print(f"[DEBUG]  Reached maximum attempts limit ({max_attempts}). Stopping.")
                    break
                    
                query = query_dict['query']
                print(f"[DEBUG] Searching with query: {query} (Attempt {total_attempts + 1}/{max_attempts})")
                queries_tried.append(query)
                total_attempts += 1
                print(f"[DEBUG] Total attempts: {total_attempts}")
                
                # Retrieve
                document_list = news_searcher.search(query=query, k=args.k)
                retrieval_result = []
                for doc in document_list:
                    one_doc = {'page_content': doc.page_content, 'metadata': doc.metadata}
                    retrieval_result.append(one_doc)
                
                # Store retrieval result for retry analysis
                all_retrieval_results.append(retrieval_result)
                
                # Get retrieved sources (full source dictionaries)
                retrieved_sources = get_retrieved_sources(retrieval_result)

                # Check if target source is in the retrieval result
                for i, retrieved_source in enumerate(retrieved_sources, start=1):
                    # Compare full dictionaries
                    if retrieved_source == target:
                        target_found = True
                        rank = i
                        target_query = query  # Store the successful query
                        filter_set = retrieval_result  # Store the full retrieval result (all k sources)
                        print(f"[DEBUG] ✅ Target source found at rank {rank}!")
                        break
                
                if target_found:
                    break

            while not target_found and queries_tried and total_attempts < max_attempts:
                remaining_attempts = max_attempts - total_attempts
                print(f"[DEBUG] Target source not found after {len(queries_tried)} queries. Generating retry queries... ({remaining_attempts} attempts remaining)")
                
                query_to_sources = get_all_retrieved_sources(queries_tried, all_retrieval_results)
                
                retry_prompt = create_query_generation_retry_prompt(
                    priors, article, starting_query, target, queries_tried, query_to_sources
                )
                
                # Call LLM to generate new queries
                print("[DEBUG] Generating retry queries with LLM...")
                messages = [{"role": "user", "content": retry_prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    retry_outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)

                retry_response = tokenizer.decode(retry_outputs[0], skip_special_tokens=False)
                retry_response = extract_response(retry_response, llm_model_name)
                
                print("\n[DEBUG] Retry Response:")
                print(retry_response)
                
                # Parse retry response
                retry_parsed = extract_json_from_response(retry_response)
                
                if retry_parsed is None:
                    print(f"[DEBUG] Could not extract JSON from retry response")
                    print(f"[DEBUG] Retry response was: {retry_response[:500]}...")
                    # Try to continue with empty queries instead of breaking
                    retry_queries = []
                else:
                    retry_queries = retry_parsed.get('queries', [])
                
                if retry_queries:
                    print(f"[DEBUG] Generated {len(retry_queries)} retry queries. Trying them now...")
                    
                    # Try retry queries (up to remaining attempts)
                    for retry_query_dict in retry_queries:
                        # Check if we've reached max attempts
                        if total_attempts >= max_attempts:
                            print(f"[DEBUG] Reached maximum attempts limit ({max_attempts}). Stopping.")
                            break
                            
                        retry_query = retry_query_dict.get('query', '')
                        if not retry_query:
                            continue
                            
                        print(f"[DEBUG] Searching with retry query: {retry_query} (Attempt {total_attempts + 1}/{max_attempts})")
                        queries_tried.append(retry_query)  # Add to queries_tried for next retry generation
                        total_attempts += 1
                        
                        # Retrieve
                        document_list = news_searcher.search(query=retry_query, k=args.k)
                        retrieval_result = []
                        for doc in document_list:
                            one_doc = {'page_content': doc.page_content, 'metadata': doc.metadata}
                            retrieval_result.append(one_doc)

                        # Store retrieval result for next retry analysis
                        all_retrieval_results.append(retrieval_result)
                        
                        # Get retrieved sources
                        retrieved_sources = get_retrieved_sources(retrieval_result)

                        # Check if target source is in the retrieval result
                        for i, retrieved_source in enumerate(retrieved_sources, start=1):
                            if retrieved_source == target:
                                target_found = True
                                rank = i
                                target_query = retry_query  # Store the successful query
                                filter_set = retrieval_result  # Store the full retrieval result (all k sources)
                                print(f"[DEBUG] Target source found at rank {rank} with retry query!")
                                break
                        
                        if target_found:
                            break
                    
                    # Continue the while loop to generate more retry queries if target not found
                    if not target_found:
                        if total_attempts >= max_attempts:
                            print(f"[DEBUG] Target source not found after {total_attempts} attempts (max limit reached)")
                        else:
                            print(f"[DEBUG] Target source still not found after retry queries. Generating new retry queries...")
                        # Continue the while loop
                        continue
                else:
                    print("[DEBUG] No retry queries generated from LLM response")
                    # Don't break - continue to try generating more queries
                    # Only break if we've exhausted attempts or hit max
                    if total_attempts >= max_attempts:
                        break
                    continue
        
            # Final check
            if not target_found and total_attempts >= max_attempts:
                print(f"[DEBUG] Target source not found after {total_attempts} attempts (max limit: {max_attempts})")
            
            # Create enhanced datapoint with new fields
            enhanced_datapoint = datapoint.copy()
            enhanced_datapoint['attempted_queries'] = queries_tried
            enhanced_datapoint['target_query'] = target_query  # None if not found
            enhanced_datapoint['filter_set'] = filter_set  # None if not found
            enhanced_dataset.append(enhanced_datapoint)
            
            # Create simplified entry with just key fields
            simplified_entry = {
                'attempted_queries': queries_tried,
                'target_query': target_query,  # None if not found
                'filter_set': filter_set,  # None if not found
                'number_of_priors': len(priors),
                'total_attempts': total_attempts
            }
            simplified_dataset.append(simplified_entry)
            
            print(f"[DEBUG] Processed datapoint {idx}. Target found: {target_found}, Target query: {target_query}")
        
        except Exception as e:
            print(f"[ERROR] Failed to process datapoint {idx}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Continuing with next datapoint...")
            # Create empty entry to maintain dataset structure
            enhanced_datapoint = datapoint.copy()
            enhanced_datapoint['attempted_queries'] = []
            enhanced_datapoint['target_query'] = None
            enhanced_datapoint['filter_set'] = None
            enhanced_dataset.append(enhanced_datapoint)
            
            simplified_entry = {
                'attempted_queries': [],
                'target_query': None,
                'filter_set': None,
                'number_of_priors': len(datapoint.get('prior_sources', [])),
                'total_attempts': 0
            }
            simplified_dataset.append(simplified_entry)
            continue
    
    # Save enhanced dataset after processing all datapoints
    output_file_path = os.path.join(proj_root, 'results', 'raw', 'no_anchor_quer_response_1.json')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
    print(f"\n[DEBUG] Enhanced dataset saved to: {output_file_path}")
    print(f"[DEBUG] Total datapoints: {len(enhanced_dataset)}")
    
    # Save simplified dataset
    simplified_output_path = os.path.join(proj_root, 'results', 'simplified', 'simplified_no_anchor_quer_response_1.json')
    os.makedirs(os.path.dirname(simplified_output_path), exist_ok=True)
    with open(simplified_output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_dataset, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Simplified dataset saved to: {simplified_output_path}")
    print(f"[DEBUG] Total simplified entries: {len(simplified_dataset)}")
    

    # save stats
    stats_output_path = os.path.join(proj_root, 'results', 'stats','stats_no_anchor_quer_response_1.json')
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
    with open(simplified_output_path, 'r', encoding='utf-8') as f:
        simplified_dataset = json.load(f)
        total = 0
        found = 0
        for entry in simplified_dataset:
            total += entry['total_attempts']
            if entry['target_query'] is not None:
                found += 1
        
        stats = {
            'total': total,
            'found': found,
            'percentage': found / total if total > 0 else 0.0,
        }
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Stats saved to: {stats_output_path}")
    print(f"[DEBUG] Total: {total}")
    print(f"[DEBUG] Found: {found}")
    print(f"[DEBUG] Percentage: {stats['percentage']}")

    
if __name__ == "__main__":
    main()

