import json


def create_promp_text(press_release, starting_query, formatted_priors):

    query_prompt = f"""
You are the Lead Research Assistant for an investigative journalist.
Your goal is to formulate a **follow-up search query** to dig deeper into a story, given the original press release, the core reporting question, and the information found so far.

### Input Context
1. **Original Press Article (what sparked the story):**
{press_release}

2. **Journalist's Core Reporting Question:**
"{starting_query}"

3. **Information Already Found (Priors):**
{formatted_priors}

### Task
- Analyze the information already found in light of the core reporting question.
- Decide what the reporter would naturally investigate next based on what is still unclear, missing, or underdeveloped.
- Use the priors to understand which angles are already covered and which directions still feel open.

### Guidelines for the Query
- **Do not** simply repeat or lightly rephrase the core reporting question.
- Think like a reporter expanding a source list: the query can seek background, data, expert analysis, official statements, historical context, industry response, or opposing viewpoints.
- Avoid queries that assume the conclusion or name the exact answer in advance.
- Keep the query under **20 words**.

### Output Format
You must output **strictly valid JSON** in the following format (no markdown code blocks, no intro text):

[
  {{
    "query": "[STRING: query for the next follow-up search on this story]"
  }}
]
"""

    return query_prompt

    


def main():
    sft_data_fp = '/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_engine_sft_data_ready.json'
    output_fp = '/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_engine_sft_formatted.json'
    
    with open(sft_data_fp, 'r') as f:
        sft_data = json.load(f)
    
    non_dict_items = 0
    non_dict_article = 0
    non_dict_starting_query = 0
    missing_press_release = 0
    missing_starting_query = 0
    missing_formatted_priors = 0
    total_missing = 0
    missing_target_query = 0
    missing_starting_query_index = 0
    index = 0
    
    # List to store formatted SFT data
    formatted_sft_data = []

    for item in sft_data:
        index += 1
        if not isinstance(item, dict):
            non_dict_items += 1
            total_missing += 1
            continue

        article = item.get('article')
        if not isinstance(article, dict):
            non_dict_article += 1
            missing_press_release += 1
            total_missing += 1
            continue

        if article.get('press_release_text') is None:
            missing_press_release += 1
            total_missing += 1
            continue

        starting_query = item.get('starting_query')
        if not isinstance(starting_query, dict):
            non_dict_starting_query += 1
            missing_starting_query += 1
            total_missing += 1
            continue

        if starting_query.get('model_output') is None:
            missing_starting_query += 1
            total_missing += 1
            missing_starting_query_index = index
            continue

        if item.get('prior_sources') is None:
            missing_formatted_priors += 1
            total_missing += 1
            continue
        
        if item.get('target_query') is None:
            missing_target_query += 1
            total_missing += 1
            continue
        
        # Extract data and create prompt
        press_release = item['article']['press_release_text']
        starting_query_text = item['starting_query']['model_output']
        formatted_priors = item['prior_sources']
        prompt_text = create_promp_text(press_release, starting_query_text, formatted_priors)
        output_text = item['target_query']
        
        # Create SFT format entry (prompt-completion format)
        sft_entry = {
            "prompt": prompt_text,
            "completion": output_text
        }
        
        formatted_sft_data.append(sft_entry)
    
    # Save formatted data
    with open(output_fp, 'w', encoding='utf-8') as f:
        json.dump(formatted_sft_data, f, indent=2, ensure_ascii=False)
    
    print(f"Non-dict items: {non_dict_items}")
    print(f"Non-dict article field: {non_dict_article}")
    print(f"Non-dict starting_query field: {non_dict_starting_query}")
    print(f"Missing press release: {missing_press_release}")
    print(f"Missing starting query: {missing_starting_query}")
    print(f"Missing formatted priors: {missing_formatted_priors}")
    print(f"Missing target query: {missing_target_query}")
    print(f"Total missing: {total_missing}")
    print(f"\nSuccessfully formatted {len(formatted_sft_data)} items")
    print(f"Saved to: {output_fp}")



if __name__ == "__main__":
    main()