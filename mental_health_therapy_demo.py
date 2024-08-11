!pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

#---------------------------------------------------------------------
!pip install bitsandbytes peft accelerate huggingface_hub

#---------------------------------------------------------------------
from huggingface_hub import notebook_login
notebook_login()


#---------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

model_id = "fadodr/finetuned_mental_health_therapy_original"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

FastLanguageModel.for_inference(model)

#---------------------------------------------------------------------
input_text = "do you think a therapist can help me in my problems ?"
instruction_text = "I need your help as a mental health therapist"
prompt = f"""
### Input:
{input_text}

### Instruction:
{instruction_text}

### Response:
"""

inputs = tokenizer([ prompt ] , return_tensors = "pt")

outputs = model.generate(**inputs.to("cuda"), max_new_tokens = 512, use_cache = True)

tokenizer.batch_decode(outputs, skip_special_tokens= True)