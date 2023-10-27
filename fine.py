from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
max_length = 128
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

dataset = load_dataset("Leul78/qanda")
def map_function(example):
    question = f"#### Instruction: Categorize the text based on the sales technique used in it from one of these eight categories only:\n\nBUILDING RAPPORT\nNEEDS ASSESMENT\nCREATING URGENCY\nSOCIAL PROOF\nOVERCOMING OBJECTION\nCROSS SELLING OR UPSELLING\nVALUE BASED SELLING\nNONE\n\n ### Input: {example['text'].strip()}"
    output = f"#### Assistant: {example['category'].strip()}"
    question_encoded = tokenizer(question)
    output_encoded = tokenizer(output, max_length=max_length-1-len(question_encoded["input_ids"]), truncation=True, padding="max_length")
    output_encoded["input_ids"] = output_encoded["input_ids"] + [tokenizer.pad_token_id]
    output_encoded["attention_mask"] = output_encoded["attention_mask"] + [0]

    input_ids = question_encoded["input_ids"] + output_encoded["input_ids"]
    attention_mask = [1]*len(question_encoded["input_ids"]) + [1]*(sum(output_encoded["attention_mask"])+1) + [0]*(len(output_encoded["attention_mask"])-sum(output_encoded["attention_mask"])-1)
    labels = [input_ids[i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
    assert len(labels) == len(attention_mask) and len(attention_mask) == len(input_ids), "Labels is not the correct length"  
    return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }  
dataset = dataset["train"].map(map_function)
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
    task_type="SEQ_CLS",
    inference_mode=False,
    target_modules=lora_target_modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    optim=optim_type,
    num_train_epochs=8,
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
