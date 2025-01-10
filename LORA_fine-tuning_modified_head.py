from huggingface_hub import login
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, BatchEncoding
from peft import LoraConfig, AutoPeftModelForSequenceClassification
from trl import SFTTrainer
from tqdm import tqdm
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score


# --------------------------------- SEQUENCE CLASSIFICATION HEAD ------------------------------------------------------

HF_token = "..."     # set the private token

login(
    token=HF_token
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------------------------------------------------
# 0. Data preparation
# ---------------------------------------------------------------------------------------------------------------------
TRAIN_PATH = r"..."     # set a path to the training data file in .tsv or csv
DEV_PATH = r"..."       # set a path to the dev data file in .tsv or csv

train_df = pd.read_csv(TRAIN_PATH, sep='\t')    # separator for .tsv
dev_df = pd.read_csv(DEV_PATH, sep='\t')


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        'example_label_1': 0,
        'example_label_2': 1,
        'example_label_3': 2,
        'example_label_4': 3,
    }
    df['target'] = df['target'].map(label_map)
    return df


train_df = preprocess_data(train_df)
dev_df = preprocess_data(dev_df)
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)

# the target column must be named 'label'; it's good to name the data column as a 'text' but not necessary, rename them if needed
train_dataset = train_dataset.rename_column("example_column_name", "text")
train_dataset = train_dataset.rename_column("target", "label")
dev_dataset = dev_dataset.rename_column("example_column_name", "text")
dev_dataset = dev_dataset.rename_column("target", "label")

# Checking datasets
print("train shape", train_dataset.shape)
print(train_dataset[0])
print("train shape", dev_dataset.shape)
print(dev_dataset[0])

# ---------------------------------------------------------------------------------------------------------------------
# 1. Setting up model, tokenizer, Lora, Training Args and Trainer
# ---------------------------------------------------------------------------------------------------------------------
model_id = "Qwen/Qwen2.5-3B-Instruct"

max_seq_length = 3072   # not sure about the optimal value
num_labels = 4  # set correct number of labels
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# choose SequenceClassification Head.
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", #set it if flash_attention is available
    num_labels=num_labels,
)

# I don't know why, but when using specific head, tokenizer doesn't work for me in SFFTrainer, so manual tokenization is needed
def tokenize_function(examples: Dataset) -> BatchEncoding:
    return tokenizer(examples["text"], padding="max_length", truncation=True,
                         max_length=max_seq_length, return_tensors='pt')    # set correct column names, use appropriate number of "examples["column_x]"


train_dataset = train_dataset.map(lambda examples: tokenize_function(examples), batched=True)
dev_dataset = dev_dataset.map(lambda examples: tokenize_function(examples), batched=True)
model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="SEQ_CLS",
)

args = TrainingArguments(
    output_dir=f"{model_id}_LORA_special_head",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
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
peft_model_id = f"{model_id}_LORA_special_head"
torch.cuda.empty_cache()

model = AutoPeftModelForSequenceClassification.from_pretrained(
    peft_model_id,
    torch_dtype=torch.float16,
    num_labels=num_labels
)
model.to(device)


def get_answer(sample: Dataset) -> Tuple:
    inputs = tokenize_function(sample).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        # pred = logits.item() # if num_labels == 1
        pred = logits.argmax().item()
    return pred, sample['label']


def evaluate(dataset: Dataset) -> Dict:
    preds = []
    targets = []
    for sample in tqdm(dataset):
        y_pred, y_true = get_answer(sample)
        preds.append(y_pred)
        targets.append(y_true)

    return {"accuracy": accuracy_score(targets, preds)}


print(evaluate(dev_dataset))
