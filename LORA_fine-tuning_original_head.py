from huggingface_hub import login
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from typing import Tuple, Dict


# --------------------------------- ORIGANAL HEAD ------------------------------------------------------------------

HF_token = "..."   # set the private token

login(
    token=HF_token
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------------------------------------------------
# 0. Data preparation
# ---------------------------------------------------------------------------------------------------------------------
TRAIN_PATH = "..."  # set a path to the training data file in .xlsx
DEV_PATH = "..."    # set a path to the dev data file in .xlsx

train_df = pd.read_excel(TRAIN_PATH)
dev_df = pd.read_excel(DEV_PATH)

train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)


def create_conversation(sample: Dataset) -> Dict:
    system_message = (
                "You are a highly intelligent assistant...\n\n"
                "Instructions:\n"
                "1. Analyze sth...\n"
                "2. Description of a task..."
                "3. Condition example: Return a score of 1 if sth.. or 0 if it isn't.\n\n"
                "Key points:\n"
                "- Return ONLY the single digit, 1 or 0.\n"

            )
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""Is condition met?\n\n
                   Requirement:\n
                   {sample["requirement"]}\n\n
                    Sample data:
                    {sample['sample_data']}"""},
            {"role": "system", "content": str(sample["target"])}
        ]
    }


train_dataset = train_dataset.shuffle().select(range(len(train_dataset)))
train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.features, batched=False)
dev_dataset = dev_dataset.shuffle().select(range(len(dev_dataset)))
dev_dataset = dev_dataset.map(create_conversation, remove_columns=dev_dataset.features, batched=False)

# Checking datasets
print("train shape", train_dataset.shape)
print(train_dataset[0])
print("train shape", dev_dataset.shape)
print(dev_dataset[0])

# ---------------------------------------------------------------------------------------------------------------------
# 1. Setting up model, tokenizer, Lora, Training Args and Trainer
# ---------------------------------------------------------------------------------------------------------------------
model_id = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", #set it if flash_attention is available
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
model, tokenizer = setup_chat_format(model, tokenizer)
max_seq_length = 3072               # max sequence length for model and packing of the dataset


peft_config = LoraConfig(
    lora_alpha=128,                 # def according to paper and Sebastian Raschka post
    lora_dropout=0.05,
    r=256,                          # def according to paper and Sebastian Raschka post
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir=f"{model_id}_LORA",  # directory to save and repository id
    num_train_epochs=3,             # number of training epochs
    per_device_train_batch_size=3,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,    # use gradient checkpointing to save memory
    optim="adamw_torch_fused",      # use fused adamw optimizer
    logging_steps=10,               # log every 10 steps
    save_strategy="epoch",          # save checkpoint every epoch
    learning_rate=2e-4,             # learning rate, based on QLoRA paper
    bf16=True,                      # use bfloat16 precision
    tf32=True,                      # use tf32 precision
    max_grad_norm=0.3,              # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,              # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",   # use constant learning rate scheduler
    push_to_hub=False,               # push model to hub
    report_to="tensorboard",        # report metrics to tensorboard
)


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    }
)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------------------------------------------------
trainer.train()
trainer.save_model()

del model
del trainer
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------------------------------
# 3. Evaluating
# ---------------------------------------------------------------------------------------------------------------------
peft_model_id = f"{model_id}_LORA"

model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_id,
    torch_dtype=torch.float16,
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


def get_prediction(sample: Dataset) -> Tuple:
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=5, do_sample=True, temperature=0.1, eos_token_id=pipe.tokenizer.eos_token_id,
                   pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    return predicted_answer, sample['messages'][2]['content']


def evaluate(dataset: Dataset) -> Dict:
    preds = []
    targets = []
    for sample in tqdm(dataset):
        print(sample)
        y_pred, y_true = get_prediction(sample)
        print(f"{y_pred} : {y_true}")
        preds.append(y_pred)
        targets.append(y_true)

    results = []
    for pred, target in zip(preds, targets):
        if pred == target:
            results.append(1)
        else:
            results.append(0)
    acc = sum(results) / len(results)
    return {"accuracy": acc}


print(evaluate(dev_dataset))