# Replace <YOUR DIRECTORY NAME> with the directory of your image folder，others  does not need to be changed
# The model is from https://huggingface.co/microsoft/Florence-2-large

!pip install -U oyaml transformers einops albumentations python-dotenv

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# <MORE_DETAILED_CAPTION> No need to change. this means generating a more detailed description, and this does not need to be changed

prompt = "<MORE_DETAILED_CAPTION>"

for i in os.listdir('<YOUR DIRECTORY NAME>'+'/'):
    if i.split('.')[-1]=='txt':
        continue
    image = Image.open('<YOUR DIRECTORY NAME>'+'/'+i)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


    parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
    print(parsed_answer)
    with open('<YOUR DIRECTORY NAME>'+'/'+f"{i.split('.')[0]}.txt", "w") as f:
        f.write(parsed_answer["<MORE_DETAILED_CAPTION>"])
        f.close()

