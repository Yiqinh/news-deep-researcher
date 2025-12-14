from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"
data_path = "/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_engine_sft_final_data_cleaned_1.json"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("json", data_files=data_path, split="train")

def formatting_func(ex):
    return ex["prompt"] + ex["completion"]

training_args = SFTConfig(
    output_dir="./query_engine_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    max_length=4096,
    completion_only_loss=True,  
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

trainer.train()
