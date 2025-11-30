import argparse
from tqdm import tqdm
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from peft import LoraConfig, TaskType
from sentence_transformers import util
import torch
import torch.nn.functional as F
from datasets import Dataset
import os
import sys
import re
import json

# this file
here = os.path.dirname(os.path.abspath(__file__))
# proj root
proj_root = os.path.dirname(here)
# add proj root to path
sys.path.append(proj_root)

from src.searcher import *

news_searcher = Searcher(index_name='../faiss_index_news_sources', model_name="Qwen/Qwen3-Embedding-8B")

def get_prompt(starting_query, formatted_priors):

    retrieval_query_prompt = f"""
        You are the Lead Research Assistant for an investigative journalist.
        Your goal is to formulate a **follow-up search query** to dig deeper into a story, given the initial intent and the information found so far.

        ### Input Context
        1. **Journalist's Original Intent:** "{starting_query}"
        2. **Information Already Found (Priors):**
        {formatted_priors}

        ### Task
        Analyze the "Information Already Found" against the "Original Intent".
        Identify an **Information Gap**: What is missing? Is there a specific person, event, date, or opposing viewpoint mentioned in the priors that requires a dedicated background search?
        
        Generate a **single JSON object** containing:
        1.  `reasoning`: A 2-sentence explanation of the "Information Gap" you identified.
        2.  `query`: A specific, keyword-focused search query (max 20 words) to fill that gap.

        ### Guidelines for the Query
        - **Do not** simply repeat the original query.
        - **Do not** write a full sentence or question (e.g., avoid "What is the relationship between...").
        - **Do** use specific entities (names, organizations, places) found in the priors if they need investigation.
        - **Do** use boolean-style logic or specific keywords if clarifying a relationship.
        
        ### Output Format
        You must output **strictly valid JSON** in the following format (no markdown code blocks, no intro text):
        
        [
            {{
                "reasoning": "The priors mention 'Project Alpha' but do not specify who funded it. We need to find financial disclosures.",
                "query": "\"Project Alpha\" funding donor disclosure financial report"
            }}
        ]

    """
    
    return retrieval_query_prompt


def embedding_similarity_reward(prompts, completions, ground_truth_text, **kwargs): 

    query_pattern = re.compile(r'"query"\s*:\s*"(.*?)"', re.DOTALL)

    rewards = []
    ground_truths = kwargs['ground_truth_text']
    
    for completion, gt_text in tqdm(zip(completions, ground_truths), desc='computing cos sim rewards', total=len(completions)):
        
        match = query_pattern.search(completion)
        
        if not match:
            # Penalty if model fails to output the required format
            rewards.append(0.0) 
            continue
            
        # Extract the capture group
        query = match.group(1).strip()
        
        retrieved = news_searcher.search(query, k=10)
        retrieved_texts = [doc.page_content for doc in retrieved]
        
        if not retrieved_texts:
            rewards.append(0.0)
            continue

        retrieved_vecs_list = news_searcher.embeddings.embed_documents(retrieved_texts)
        gt_vec_list = news_searcher.embeddings.embed_query(gt_text)
        
        # Convert to PyTorch Tensors
        # Shape: (k, hidden_dim)
        retrieved_tensor = torch.tensor(retrieved_vecs_list, device="cuda") 
        # Shape: (1, hidden_dim)
        gt_tensor = torch.tensor(gt_vec_list, device="cpu").unsqueeze(0) 

        # 4. Compute Similarity for EACH document
        # We use standard Torch Cosine Similarity with broadcasting
        # gt_tensor (1, D) vs retrieved_tensor (k, D) -> scores (k,)
        scores = F.cosine_similarity(gt_tensor, retrieved_tensor, dim=1)
        
        # scores is now a tensor, e.g., [0.85, 0.60, 0.45, 0.10]
        
        final_reward = scores.mean().item()     
        rewards.append(final_reward)

    return rewards

def create_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    for entry in raw_data:
        target = entry.get('target_source', {})
        embed_string = ""
        for k, v in target.items():
            embed_string += v
            embed_string += '\n'
        
        processed_data.append({
            "prompt": get_prompt(entry['starting_query']['model_output'], entry.get('prior_sources', [])), 
            "ground_truth_text": embed_string,
        })

    # Create the HF Dataset
    hf_dataset = Dataset.from_list(processed_data)
    return hf_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="huggingface model path")
    args = parser.parse_args()

    dataset = create_dataset("expanded_train_10000.json")

    peft_config = LoraConfig(
        r=16,                       # Rank
        lora_alpha=32,              # Alpha (scaling factor)
        lora_dropout=0.05,          # Dropout
        target_modules=[
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "gate_proj", "up_proj", 
            "down_proj"
        ],
        task_type="CAUSAL_LM",
        bias="none"
    )

    training_args = GRPOConfig(
        output_dir="output/Qwen-GRPO-LoRA",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,           
        max_prompt_length=4096,
        max_completion_length=4096,
        learning_rate=5e-6,
        logging_steps=10,
        fp16=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=embedding_similarity_reward,
        train_dataset=dataset, # Contains column "ground_truth_text"
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__=="__main__":
    main()