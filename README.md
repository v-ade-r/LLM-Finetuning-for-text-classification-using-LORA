# LLM-Finetuning-using-LORA

This is the Schematic Blueprint for Finetuning LLM (e.g. Qwen or Llama) for semantic text classification using LORA. 

## LORA - theory
LoRA (Low-Rank Adaptation), introduced by Hu et al., optimizes model fine-tuning by decomposing weight updates, ΔW, into a low-rank representation. Typically during backpropagation model learns ΔW matrix containing updates for the original weights which minimize the loss function during training.

W(updated) = W + ΔW

Instead of explicitly computing ΔW, LoRA directly learns two smaller matrices, A and B, during training. Decomposition can be described as ΔW = AB (Matrix multiplication of A and B gives ΔW because, A has the same number of rows as ΔW and B has the same number of columns as ΔW). The key hiperparameter is r - rank. If ΔW has 5,000 rows and 10,000 columns, it stores 50,000,000 parameters. If we choose A and B with r=8, then A has 5,000 rows and 8 columns, and B has 8 rows and 10,000 columns, that's 5,000×8 + 8×10,000 = 120,000 parameters, which is about 416× less than 50,000,000.

Original weights during the training are untouched. A and B are only trained. After training AB are multiplicated creating matrix with the same dimensions as the original weights matrix. This matrix is then added to the original weights matrix.
Here, A and B are much smaller in size compared to ΔW, significantly reducing memory and computational overhead. This efficient approach makes LoRA ideal for adapting large models to new tasks while maintaining scalability and performance.

## **Justification of this code**
While this approach has been widely adopted, it often requires some tweaks for custom data which is not in the dataset format. Here, you only need to provide the file path and define a prompt tailored to your problem and data. Additionally, this guide outlines two approaches: the first utilizes the model's original head, while the second introduces a modified head with label mapping.

## **Usage tips**
### I. LORA with model's original head
  1. Set your private Hugginface token.
  2. Set a path to your train and dev data files in .xlsx (create DataFrame from whatever format file you like)
  3. Create a system message tailored to your task.
    a) Describe the role e.g. Assistant experienced in evaluating geographical locations of houses.
    b) Describe the task e.g. Your task is to recognize whether a list of locations contains at least one object matching specific geographical requirements.
    c) Describe the condition e.g Return a score of 1 if a list of locations contains at least one object matching specific geographical requirements or 0 otherwise.
  4. Create a prompt template according to your problem
     a) "Is condition met?" - Swap this for your condition e.g. Is in Sample data at least one suitable object matching geographical requirement?
     b) "{sample["requirement"]}" - replace 'requirement' with the column name containing the requirements. (e.g desert, in mountains, with population over 1kk, on hte seashore...)
     c) "{sample['sample_data']}" - replace "sample data" with the column name containing your data (e.g [Boston, Chicago, New York, Nevada, Aspen])
  5. Select an open source model having in mind GPU RAM limitations (3B model is roughly the biggest possible for 8GB single GPU)
  6. Set training parameters. These are optimal according to Sebastian Raschka. You probably need to reduce 'lora_alpha' and 'r' numbers (possibly also 'num_train_epochs',
    'per_device_train_batch_size'), to be able to train smoothly on smaller GPUs.
  7. Start training, save the model, evaluate and done!


### II. LORA with model's modified head
  1. Set your private Hugginface token.
  2. Set a path to your train and dev data files in .tsv or .csv (create DataFrame from whatever format file you like)
  3. Map possible semantic labels to integers.
  4. Rename the column containing labels to 'label'!!!
  5. Set num_labels variable accordingly!!!!
  6. In tokenize_function use as many "examples["custom_column_name"]" objects as as you have input columns (e.g. if comparing 2 sentences from 2 columns, then use examples["column1_name"], examples["column2_name"])
  7. Select an open source model having in mind GPU RAM limitations (3B model is roughly the biggest possible for 8GB single GPU)
  8. Set training parameters. These are optimal according to Sebastian Raschka. You probably need to reduce 'lora_alpha' and 'r' numbers (possibly also 'num_train_epochs',
    'per_device_train_batch_size'), to be able to train smoothly on smaller GPUs.
  9. Start training, save the model, evaluate and done!

## References
https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

https://www.philschmid.de/fine-tune-llms-in-2024-with-trl?WT.mc_id=academic-105485-koreyst
