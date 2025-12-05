import os
import json


def remove_incomplete_datapoints(data_file_path):

    incomplete_datapoints = 0
    no_press_release_text = 0

    relevant_fields = ['article', 'starting_query', 'target_query']

    with open(data_file_path, 'r') as f:
        data = json.load(f)

    for item in data:
        for field in relevant_fields:
            if field == 'article':
                if not item[field].get('press_release_text'):
                    no_press_release_text += 1
            if item[field] == None:
                incomplete_datapoints += 1
                break

    return [incomplete_datapoints, no_press_release_text]

def main():
    
    sft_data_fp = '/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_and_filter_sft_data_1.json'

    if not os.path.exists(sft_data_fp):
        print(f"File not found: {sft_data_fp}")
        return 
    
    incomplete_datapoints, no_press_release_text = remove_incomplete_datapoints(sft_data_fp)
    print(f"Incomplete datapoints: {incomplete_datapoints}")
    print(f"No press release text: {no_press_release_text}")

if __name__ == "__main__":
    main()