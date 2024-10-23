# import necessary packages
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Use a pipeline as a high-level helper
from huggingface_hub import InferenceClient

# load data
dataset_path = "/var/scratch/mst437/"
#dataset_path = "C:/Users/milen/Desktop/Stage OCW/Code/Data/"

test_df = pd.read_csv(dataset_path + 'test_df.csv')
new_test_df = pd.read_csv(dataset_path + 'new_test_df.csv')

# model direct inladen
# Load model directly
#model_path = "C:/Users/milen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B"
#model_path = "./models--meta-llama--Meta-Llama-3-8B/"
model_path = "/var/scratch/mst437/Llama3"
#model_path = "C:/Users/milen/Desktop/Llama2" 

model_name = 'llama3'

if model_name == 'llama2':
	save_directory = "/var/scratch/mst437/Llama2" #replace with your path to the llama2 model
elif model_name == 'llama3':
	save_directory =  "/var/scratch/mst437/Llama3" #replace with your path to the llama3 model
      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model_llama = AutoModelForCausalLM.from_pretrained(save_directory, torch_dtype=torch.float16, low_cpu_mem_usage=True)

model_llama.to(device)
#tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True) 
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")
#model_llama = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True) 
#model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")

client = InferenceClient(model_llama)

# get token IDs for '0' and '1'
token_0_id = tokenizer.convert_tokens_to_ids('0')
token_1_id = tokenizer.convert_tokens_to_ids('1')
allowed_token_ids = [token_0_id, token_1_id]

item = "de migratiestroom loopt uit de hand."

#input_prompt = f"Je bent een expert op het gebied van bias, stereotypering en discriminatie. Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen, OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat. Geef de volgende zin een label, antwoord alleen met 0 of 1. Geef de zin het label 0 voor niet-biased en 1 voor biased. De zin is: '{item}'. Antwoord alleen met '0' of '1'."
input_prompt = f"""
      <s>[INST] <<SYS>>
      Geef de volgende zin een label, antwoord alleen met biased of niet-biased.<</SYS>>
      De zin: {item}[/INST]"""

#input_prompt = f"""
#        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#        Geef de volgende zin een label, antwoord alleen met 0 of 1. Geef de zin het label 0 voor niet-biased en 1 voor biased<|eot_id|>
#        <|start_header_id|>user<|end_header_id|>
#        De zin: {item}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#        """
  
#input_prompt =  ". De zin is: '{item}'. Antwoord alleen met '0' of '1'. Geef geen verdere uitleg."

#prompt the model
inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
input_token_len = input_ids.shape[-1]
     
#prompt the model with temperature near 0 to produce deterministica responses
outputs = model_llama.generate(
      input_ids,
      attention_mask= attention_mask,
      pad_token_id= 128001,
      max_new_tokens=50,
      num_return_sequences=1,
      temperature=0.0000001,
      do_sample = False,
      force_words_ids=[[token_0_id], [token_1_id]],  # Only allow '0' or '1'
      num_beams = 2,
      output_scores=True,
      return_dict_in_generate=True,
      )
     
#extract the generated text
generated_text = tokenizer.decode(outputs.sequences[0][input_token_len:], skip_special_tokens=False)
#predictions.append(generated_text)
print(generated_text)

# loop om voorspellingen op te halen en sla ze op als lijst
# loop to get predictions and save them in a list
predictions = []

for item in test_df['text']:
  torch.cuda.empty_cache()
  input_prompt = f"""
      <s>[INST] <<SYS>>
      Geef de volgende zin een label, antwoord alleen met 'biased' of 'niet-biased'.<</SYS>>
      De zin: {item}[/INST]"""

  #prompt the model
  inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
  input_ids = inputs['input_ids'].to(device)
  attention_mask = inputs['attention_mask'].to(device)
  input_token_len = input_ids.shape[-1]
     
  #prompt the model with temperature near 0 to produce deterministica responses
  outputs = model_llama.generate(
      input_ids,
      attention_mask= attention_mask,
      pad_token_id= 128001,
      max_new_tokens=50,
      num_return_sequences=1,
      temperature=0.0000001,
      do_sample = False,
      force_words_ids=[[token_0_id], [token_1_id]],  # Only allow '0' or '1'
      num_beams = 2,
      output_scores=True,
      return_dict_in_generate=True,
      )
     
  #extract the generated text
  generated_text = tokenizer.decode(outputs.sequences[0][input_token_len:], skip_special_tokens=False)
  if generated_text[-1] == '0':
      predictions.append(0)
  elif generated_text[-1] == '1':
      predictions.append(1)
  else:
      predictions.append(2) # no prediction is incorrect prediction

print(predictions)
print(list(test_df['label']))

def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='macro')

f1 = f1_multiclass(list(test_df['label']), predictions)
print(f1)

# herhaal voor ongeziene woorden
# repeat for unseen words
new_predictions = []

for item in new_test_df['text'][0]:
  #input_prompt = f"Je bent een expert op het gebied van bias, stereotypering en discriminatie. Je taak is om zinnen te classificeren volgens de volgende definitie: een zin bevat bias als de zin een stereotype, uitsluiting, machtsverhoudingen of vooroordelen bevat, die impliciet of expliciet kunnen zijn, en die gaan over een af te bakenen groep personen, OF als een zin een woord bevat wat moet worden vermeden omdat het bias bevat. Geef de volgende zin een label, antwoord alleen met 0 of 1. Geef de zin het label 0 voor niet-biased en 1 voor biased. De zin is: '{item}'. Antwoord alleen met '0' of '1'."
  torch.cuda.empty_cache()
  input_prompt = f"""
      <s>[INST] <<SYS>>
      Geef de volgende zin een label, antwoord alleen met 'biased' of 'niet-biased'.<</SYS>>
      De zin: {item}[/INST]"""

  #prompt the model
  inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
  input_ids = inputs['input_ids'].to(device)
  attention_mask = inputs['attention_mask'].to(device)
  input_token_len = input_ids.shape[-1]
     
  #prompt the model with temperature near 0 to produce deterministica responses
  outputs = model_llama.generate(
      input_ids,
      attention_mask= attention_mask,
      pad_token_id= 128001,
      max_new_tokens=50,
      num_return_sequences=1,
      temperature=0.0000001,
      do_sample = False,
      force_words_ids=[[token_0_id], [token_1_id]],  # Only allow '0' or '1'
      num_beams = 2,
      output_scores=True,
      return_dict_in_generate=True,
      )
     
  #extract the generated text
  generated_text = tokenizer.decode(outputs.sequences[0][input_token_len:], skip_special_tokens=False)
  if generated_text[-1] == '0':
      new_predictions.append(0)
  elif generated_text[-1] == '1':
      new_predictions.append(1)
  else:
      new_predictions.append(2) # no prediction is incorrect prediction

print(new_predictions)

def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='macro')

labels_int_new = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

f1_new = f1_multiclass(labels_int_new, new_predictions)
print(f1_new)

#print("llama-3 predictions:", predictions)


