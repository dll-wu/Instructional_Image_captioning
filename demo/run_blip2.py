import requests
import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "Salesforce/blip2-opt-2.7b"
lora_model_path = "kurileo/blip2-opt-2.7b-refines"

base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    device_map="auto",
    torch_dtype=torch.bfloat16
    )
base_processor = AutoProcessor.from_pretrained(base_model_name)
config = PeftConfig.from_pretrained(lora_model_path)
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = base_processor(images=image, return_tensors="pt").to(device)
outputs = lora_model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    )

generated_text = base_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
