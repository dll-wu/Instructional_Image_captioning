import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "kurileo/git-base-refines"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
base_processor = AutoProcessor.from_pretrained("microsoft/git-base")

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = base_processor(images=image, return_tensors="pt").to(device)
outputs = base_model.generate(
    **inputs,
    max_length=1024,
    repetition_penalty=1.5,
    )

generated_text = base_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
