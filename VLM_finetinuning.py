#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

os.system('pip install --upgrade pip')
os.system('pip install -q accelerate -U')
os.system('pip install -q bitsandbytes -U')
os.system('pip install -q trl -U')
os.system('pip install -q peft -U')
os.system('pip install -q transformers -U')
os.system('pip install -q datasets -U')
os.system('pip install qwen-vl-utils')


# In[2]:


system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


# In[3]:


from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
dataset= load_dataset(dataset_id)


# In[4]:


dataset


# In[5]:


import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig,AutoModelForCausalLM,TrainingArguments

torch.cuda.set_device(0)
model_id = "Qwen/Qwen2-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map={"": 0},  # Explicitly map to GPU 0
    quantization_config=bnb_config,
).to(device)

processor = Qwen2VLProcessor.from_pretrained(model_id)


# In[7]:


from qwen_vl_utils import process_vision_info

def collate_fn(data):
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data["image"],
                },
                {
                    "type": "text",
                    "text": data["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": data["label"][0]}],
        },
    ]
    data["text"] = processor.apply_chat_template(message, tokenize=False)
    image_inputs = [process_vision_info(message)[0]]
    data = processor(text=data["text"],images=image_inputs,return_tensors="pt",padding=True)
    labels = data["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    data["labels"] = labels
    return data


# In[8]:


formatted_dataset = dataset["test"].select(range(50)).map(collate_fn)
formatted_dataset_test = dataset["val"].select(range(100)).map(collate_fn)


# In[9]:


from qwen_vl_utils import process_vision_info

def collate_fn_new(batch):
    batch_texts = []
    batch_images = []

    for data in batch:  # Iterate over batch items
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": data["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": data["label"][0]}],
            },
        ]
        
        text = processor.apply_chat_template(message, tokenize=False)
        image_input = process_vision_info(message)[0]
        
        batch_texts.append(text)
        batch_images.append(image_input)

    # Process batch using the processor
    processed_data = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )

    labels = processed_data["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handling Image Token IDs for Qwen2VLProcessor
    if isinstance(processor, Qwen2VLProcessor):  
        image_tokens = [151652, 151653, 151655]  
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    processed_data["labels"] = labels
    return processed_data


# In[10]:


formatted_dataset


# In[11]:


from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config).to(device)

# Print trainable parameters
peft_model.print_trainable_parameters()


# In[12]:


from trl import SFTConfig

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=10,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=10, # evaluate every 10 steps
    logging_steps=1,
    logging_strategy="steps",     # Log at steps instead of silent mode
    gradient_checkpointing=True, # recomputes forward pass activations in backward pass to save memory
    save_steps=500 # checkpoint every 500 steps
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset


# In[13]:


from trl import SFTTrainer

from datasets import Dataset

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_dataset_test,
    data_collator=collate_fn_new,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)


# In[14]:


# ✅ Check dataset before training
print(f"Training dataset size: {len(trainer.train_dataset)}")
print(f"Eval dataset size: {len(trainer.eval_dataset)}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# ✅ Run a single batch manually to verify dataset & model
print("Testing a single batch forward pass...")
try:
    batch = next(iter(trainer.get_train_dataloader()))
    print("Batch keys:", batch.keys())  # Check if inputs are correctly formatted
    outputs = model(**batch)
    print("Single batch forward pass success.")
except Exception as e:
    print("Error in single batch forward pass:", e)
print("Starting training...")
trainer.train(resume_from_checkpoint=False)

# ✅ Print trainer state
print("Trainer state after training:")
print(trainer.state)


# In[ ]:


trainer.train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




