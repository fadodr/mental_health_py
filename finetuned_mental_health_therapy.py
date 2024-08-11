!pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

#------------------------------------------------------------------------------------
!pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install huggingface_hub datasets

#---------------------------------------------------------------------
from huggingface_hub import notebook_login
notebook_login()

#---------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch
import os

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True
)

#---------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

#-------------------------------------------------------------------------
prompt = """Below is an input that describes a task, paired with an instruction that provides further context. Write a response that appropriately completes the request.
### Input : {}

### Instruction : {}

### Response: {}
"""

EOS_TOKEN = tokenizer.eos_token

def format_prompts(examples):
    texts = []
    inputs = examples["input"]
    instructions = examples["instruction"]
    outputs = examples["output"]
    for input, instruction, output in zip(inputs, instructions, outputs):
        # Must add EOS_TOKEN, otherwise the generation will go on forever!
        text = prompt.format(input, instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

#---------------------------------------------------------------------
from datasets import load_dataset

dataset = load_dataset("fadodr/mental_health_therapy", split = "train")

dataset = dataset.map(format_prompts, batched=True)

#---------------------------------------------------------------------
from transformers import TrainingArguments

HF_USERNAME = "fadodr"

output_dir = f"{HF_USERNAME}/fadodr/finetuned_mental_health_therapy"

training_argument = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        #num_train_epochs = 5
        logging_steps = 1,
        optim = "paged_adamw_32bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        max_grad_norm = 0.3,
        warmup_ratio = 0.03,
        output_dir = output_dir,
        save_steps = 200,
        push_to_hub = True,
        report_to = "tensorboard"
    )

#------------------------------
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = training_argument
)

#------------------------------
training_stats = trainer.train()