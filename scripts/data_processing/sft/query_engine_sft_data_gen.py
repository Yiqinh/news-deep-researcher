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

    with open(sft_data_fp, 'r') as f:
        sft_data = json.load(f)
    
    non_dict_items = 0
    non_dict_article = 0
    non_dict_starting_query = 0
    missing_press_release = 0
    missing_starting_query = 0
    missing_formatted_priors = 0
    total_missing = 0
    missing_starting_query_index = 0
    index = 0

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

        if starting_query.get('model_output') is None:
            missing_starting_query += 1
            total_missing += 1
            missing_starting_query_index = index
            continue

        if item.get('prior_sources') is None:
            missing_formatted_priors += 1
            total_missing += 1
            continue
        
        
        #press_release = item[0]['article']['press_release_text']
        #starting_query = item['starting_query']['model_output']
        #formatted_priors = item['prior_sources']
        #prompt_text = create_promp_text(press_release, starting_query, formatted_priors)
        #print(prompt_text)
    
    print(f"Non-dict items: {non_dict_items}")
    print(f"Non-dict article field: {non_dict_article}")
    print(f"Non-dict starting_query field: {non_dict_starting_query}")
    print(f"Missing press release: {missing_press_release}")
    print(f"Missing starting query: {missing_starting_query}")
    print(f"Missing formatted priors: {missing_formatted_priors}")
    print(f"Total missing: {total_missing}")



if __name__ == "__main__":
    main()