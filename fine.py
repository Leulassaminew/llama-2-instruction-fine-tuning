from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from functools import partial
import copy
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
max_length = 256
load_in_4bit = True
lora_alpha = 16             # How much to weigh LoRA params over pretrained params
lora_dropout = 0.1          # Dropout for LoRA weights to avoid overfitting
lora_r = 32                 # Bottleneck size between A and B matrix for LoRA params
lora_bias = "all" 
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
# Trainer params
output_dir = "outputs_squad"                              # Directory to save the model
optim_type = "adafactor"                            # Optimizer type to train with 
learning_rate = 0.00005                              # Model learning rate
weight_decay = 0.002                                # Model weight decay
per_device_train_batch_size = 6                     # Train batch size on each GPU
per_device_eval_batch_size = 6                      # Eval batch size on each GPU
gradient_accumulation_steps = 2                     # Number of steps before updating model
warmup_steps = 5                                    # Number of warmup steps for learning rate
save_steps = 100                                     # Number of steps before saving model
logging_steps = 25  
if load_in_4bit == True:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained("Leul78/llama-13b-chat-pri",
                                                trust_remote_code=True, 
                                                device_map="auto", 
                                                quantization_config=bnb_config,
                                                 token="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",
                                                cache_dir="./models",)
# Load in the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Leul78/llama-13b-chat-pri",
                                            trust_remote_code=True,
                                          token="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",
                                            cache_dir="./models",)
                                                
tokenizer.pad_token = tokenizer.eos_token
prompt_input = ("Below is an instruction that describes a task, paired with an input that provides further context."
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\nCategorize the sales technique used in the Input.\n\n### Input:\n{input}\n\n### Response:"
                  )
def get_preprocessed_samsum( tokenizer, split):
    dataset = datasets.load_dataset("Leul78/total", split=split)

    prompt = (f"""Below is an instruction that describes a task, paired with an input that provides further context."
                Write a response that appropriately completes the request.\n\n
                ### Instruction:\nCategorize the sales technique used in the Input.\n\n### Input:\n{input}\n\n### Response:"""
                  )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(input=sample["text"]),
            "response": sample["category"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["response"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
dataset = get_preprocessed_samsum(tokenizer,"train")
# Randomize data
dataset = dataset.shuffle()

# Test/train split
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
data_train = dataset.select(range(train_size))
data_test = dataset.select(range(train_size, train_size + test_size))

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias=lora_bias,
    task_type="CAUSAL_LM",
    inference_mode=False,
    target_modules=lora_target_modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    optim=optim_type,
    num_train_epochs=2,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    do_train=True,
    warmup_steps=warmup_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
