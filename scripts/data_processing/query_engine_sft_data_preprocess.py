import os
import json


def remove_incomplete_datapoints(item):
    if item == None:
        return False

    relevant_fields = ['article', 'starting_query', 'target_query']

    for field in relevant_fields:
        if item.get(field) is None:
            return False
    
    return True           

def main():
    sft_data_fp = '/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_and_filter_sft_data_1.json'
    output_fp = '/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_engine_sft_data_ready.json'

    if not os.path.exists(sft_data_fp):
        print(f"File not found: {sft_data_fp}")
        return 
    
    #Load the data first
    print(f"Loading dataset from: {sft_data_fp}")
    with open(sft_data_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Filter complete datapoints
    complete_data = []
    incomplete_data = 0
    for item in data:
        if remove_incomplete_datapoints(item):
            complete_data.append(item)
        else:
            incomplete_data += 1

    # Save filtered dataset
    print(f"Saving to: {output_fp}")
    os.makedirs(os.path.dirname(output_fp) if os.path.dirname(output_fp) else '.', exist_ok=True)
    with open(output_fp, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=4, ensure_ascii=False)
    
    print("Done, incomplete data: ", incomplete_data)
    print("Complete data: ", len(complete_data))

if __name__ == "__main__":
    main()