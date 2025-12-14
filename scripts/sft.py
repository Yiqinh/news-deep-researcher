from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

model_name = "Qwen/Qwen2.5-7B-Instruct"
data_path = "/lfs/local/0/aaronjohn/news-deep-researcher/results/raw/Query_engine_sft_final_data_cleaned_1.json"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

# Big VRAM saver for long context
model.gradient_checkpointing_enable()
model.config.use_cache = False

dataset = load_dataset("json", data_files=data_path, split="train")

training_args = SFTConfig(
    output_dir="./query_engine_sft_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=500,
    bf16=True,          
    fp16=False,
    max_length=4096,
    completion_only_loss=True,
)

# LoRA config (good default for 7B instruction SFT)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,   # TRL uses this instead of tokenizer in newer versions
    peft_config=peft_config,
)

trainer.train()

# Saves adapter to output_dir
trainer.save_model()
