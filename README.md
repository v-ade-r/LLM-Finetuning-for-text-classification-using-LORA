# LLM-Finetuning-for-text-classification-using-LORA

This is the Schematic Blueprint for LLM Finetuning for semantic text classification using LORA. 

## **Justification:**
While this approach has been widely adopted, it often requires custom tweaks for custom data which is not in the dataset format. Here, you only need to provide the file path and define a prompt tailored to your problem and data. Additionally, this guide outlines two approaches: the first utilizes the model's original head, while the second introduces a modified head with label mapping.

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
  5. Set num_labels variable accoringly!!!!
  6. In tokenize_function use as many "examples["custom_column_name"]" objects as as you have input column (e.g. if comparing 2 sentences from 2 columns, then use examples["column1_name"], examples["column2_name"])
  7. Select an open source model having in mind GPU RAM limitations (3B model is roughly the biggest possible for 8GB single GPU)
  8. Set training parameters. These are optimal according to Sebastian Raschka. You probably need to reduce 'lora_alpha' and 'r' numbers (possibly also 'num_train_epochs',
    'per_device_train_batch_size'), to be able to train smoothly on smaller GPUs.
  9. Start training, save the model, evaluate and done!

## References
https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

https://www.philschmid.de/fine-tune-llms-in-2024-with-trl?WT.mc_id=academic-105485-koreyst
