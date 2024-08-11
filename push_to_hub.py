!pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

#---------------------------------------------------------------------
!pip install transformers huggingface_hub peft

#---------------------------------------------------------------------
from huggingface_hub import login

login(token="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")

#---------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    dtype = torch.float16,
    max_seq_length = 2048,
    load_in_4bit = True
)

#---------------------------------------------------------------------
from peft import peft_model

adapter_model = "fadodr/finetuned_mental_health_therapy"

model = peft_model.PeftModel.from_pretrained(model, adapter_model)

model = model.to('cuda')
model.eval()

#---------------------------------------------------------------------
model = model.merge_and_unload()

#---------------------------------------------------------------------
new_model_id =  'fadodr/finetuned_mental_health_therapy_original'

model.push_to_hub(new_model_id, tokenizer = tokenizer)

tokenizer.push_to_hub(new_model_id)