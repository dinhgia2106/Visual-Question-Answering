import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import LlavaForConditionalGeneration
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from PIL import Image

# 2. Read test data
test_data = []
# Corrected path based on directory structure
test_set_path = os.path.join('vqa_coco_dataset', 'vaq2.0.TestImages.txt')

if not os.path.exists(test_set_path):
    print(f"Error: File not found at {test_set_path}")
    # Fallback to user provided path if the above fails, though unlikely given my list_dir
    # test_set_path = './vaq2.0.TestImages.txt' 
    exit(1)

print(f"Reading data from {test_set_path}...")
with open(test_set_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        if len(temp) < 2:
            continue
            
        # User logic: qa = temp[1].split('?')
        # Assuming temp[1] is "Question? Answer" or similar
        qa_part = temp[1]
        qa = qa_part.split('?')
        
        if len(qa) == 3:
            answer = qa[2].strip()
        elif len(qa) >= 2:
            answer = qa[1].strip()
        else:
            answer = ""
            
        # User logic: image_path = temp[0][:-2]
        # This suggests temp[0] might be "COCO_val2014_000000000001.jpg#0" or something?
        # Or maybe it's just a filename and they want to remove extension? 
        # But removing last 2 chars of .jpg gives .j which is wrong.
        # Let's assume temp[0] is the image filename or ID.
        # If it's "COCO_val2014_...jpg", [:-2] removes "pg".
        # If it's "COCO_val2014_...", maybe it has an ID suffix?
        # I'll keep the user's logic but print a warning if it looks weird.
        image_name_raw = temp[0]
        image_name = image_name_raw[:-2] 
        
        # Construct full question
        question = qa[0] + '?'
        
        data_sample = {
            'image_path': image_name,
            'question': question,
            'answer': answer
        }
        test_data.append(data_sample)

print(f"Loaded {len(test_data)} samples.")

# 3. Load Model
print("Loading LLaVA model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please ensure you have the required libraries installed: transformers, bitsandbytes, accelerate")
    exit(1)

# 4. Prompt function
def create_prompt(question):
    prompt = f"""### INSTRUCTION:
Your task is to answer the question based on the given image. You can only answer 'yes' or 'no'.
### USER: <image>
{question}
### ASSISTANT:"""
    return prompt

# 5. Generation Config
generation_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
)

# 6. Prediction
# Select a sample
idx = 0 
if len(test_data) > idx:
    sample = test_data[idx]
    question = sample['question']
    image_name = sample['image_path']
    
    # User path: val2014-resised
    # My check: vqa_coco_dataset/val2014-resised
    image_dir = os.path.join('vqa_coco_dataset', 'val2014-resised')
    image_path = os.path.join(image_dir, image_name)
    
    print(f"Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        # Try adding .jpg if missing
        if os.path.exists(image_path + ".jpg"):
            image_path += ".jpg"
            print(f"Found with .jpg extension: {image_path}")
        else:
            print("Could not find image file.")
            # List dir to see what's there if failed (for debugging)
            # print(os.listdir(image_dir)[:5])
    
    if os.path.exists(image_path):
        label = sample['answer']
        try:
            image = Image.open(image_path)
            
            prompt = create_prompt(question)
            inputs = processor(prompt, image, return_tensors="pt").to(device)

            print("Generating response...")
            output = model.generate(**inputs, generation_config=generation_config)
            generated_text = processor.decode(output[0], skip_special_tokens=True)

            print("-" * 20)
            print(f"Question: {question}")
            print(f"Label: {label}")
            # Robust splitting
            if '### ASSISTANT:' in generated_text:
                prediction = generated_text.split('### ASSISTANT:')[1].strip()
            else:
                prediction = generated_text
            print(f"Prediction: {prediction}")
            print("-" * 20)

            # Optional: Show image
            # plt.imshow(image)
            # plt.axis("off")
            # plt.show()
            
        except Exception as e:
            print(f"Error during inference: {e}")
else:
    print("No test data found.")
