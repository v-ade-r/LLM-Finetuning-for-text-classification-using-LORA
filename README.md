# LLM-Finetuning-for-text-classification-using-LORA

This is the Schematic Blueprint for LLM Finetuning for semantic text classification using LORA. 

**Justification:**
Although there are many examples of this approach being used, usually there are some tweaks needed to be made for custom data not in the dataset format. Essentialy you only need to provide a path to files, and fill in the prompt matching your problem and data.
Moreover I present the blueprint for a prompt structure in 1st approach and for the use of different model head and label mapping in 2nd approach.

**Usage tips:**
I. LORA with model's original head
  1. Set your private Hugginface token.
  2. Set a path for your train and dev data files in .xlsx (create DataFrame from whatever format file you like)
  3. Create a system message according to your problem.
    a) Describe the role e.g. Assistant experienced in inspecting geographical location of the houses.
    b) Describe the task e.g. Your task is to recognize if in the list of locations, there is at least one suitable object matching geographical requirements.
    c) Describe the condition e.g Return a score of 1 if there is at least one one suitable object matching geographical requirements or 0 if it isn't.
  4. Create a prompt template according to your problem
     a) "Is condition met?" - Swap this for your condition e.g. Is in Sample data at least one suitable object matching geographical requirement?
     b) "{sample["requirement"]}" - swap requirement for the column name when you have requirements e.g desert, in mountains, with population over 1kk, on hte seashore...
     c) "{sample['sample_data']}" - swap "sample data" for the column name when you have your data listed e.g [Boston, Chicago, New York, Nevada, Aspen]
  5. Choose the open source model, remembering about GPU RAM limitations (3B model is roughly the biggest possible for 8GB single GPU)
  6. Set training parameters. These are optimal according to Sebastian Raschka. You probably need to reduce 'lora_alpha' and 'r' numbers (possibly also 'num_train_epochs',
    'per_device_train_batch_size'), to be able to train smoothly on smaller GPUs.
  7. Start training, save the model, evaluate and done!


II. LORA with model's changed head.
  1. Set your private Hugginface token.
  2. Set a path for your train and dev data files in .tsv or .csv (create DataFrame from whatever format file you like)
  3. Map your semantic labels to integers.
  4. Rename the column where you have labels to 'label'!!!
  5. Set num_labels variable accoringly!!!!
  6. In tokenize_function use as many "examples["custom_column_name"]" objects as your number of columns (e.g. maybe you are comparing 2 sentences from 2 columns, then use examples["column1_name"], examples["column2_name"])
  7. Choose the open source model, remembering about GPU RAM limitations (3B model is roughly the biggest possible for 8GB single GPU)
  8. Set training parameters. These are optimal according to Sebastian Raschka. You probably need to reduce 'lora_alpha' and 'r' numbers (possibly also 'num_train_epochs',
    'per_device_train_batch_size'), to be able to train smoothly on smaller GPUs.
  9. If you want Set output_dir in TrainingArguments which will be also peft_model_id later in evaluation.
  10. Start training, save the model, evaluate and done!
