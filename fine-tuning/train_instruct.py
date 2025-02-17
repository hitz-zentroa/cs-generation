# INITIAL IMPORTS
import numpy as np

import wandb

import torch

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import argparse

from utils.load_parallel import *

# Tokenization function


# # Preprocessing function to tokenize 'text' (input) and 'label' (output)
# # Tokenization Function
# def preprocess_function(examples):
#     inputs = examples["EN"]
#     targets = examples["CS"]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
#     # Setup the tokenizer for targets
#     labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length").input_ids
#     model_inputs["labels"] = labels
#     return model_inputs





# # Compute metrics function
# def compute_metrics(eval_results):
#     """
#     Function to compute all metrics
#     """
#     perplexity = math.exp(eval_results['eval_loss'])
#     return {"perplexity":perplexity}


def train_formatting_function(data):
    """
    Template for training
    """
    full_text = []
    
    for i in range(len(data['CS'])):


        # formated_sen_instruct = [
        #     {"role": "user", "content":f"Provide a brief counter-narrative in response to the user's hate speech. Ensure the output does not contain line breaks.{data['HS'][i]}"},
        #     {"role": "assistant", "content":data['CN'][i]}
        #     ]

        formated_sen_chat = [
        {"role": "system", "content": "You are a bilingual speaker of English and Spanish. Translate the following English sentence into code-switched text between both languages:"},
        {"role": "user", "content":data['EN'][i]},
        {"role": "assistant", "content":data['CS'][i]},
        ]

        formated_sen = tokenizer.apply_chat_template(formated_sen_chat, add_generation_prompt=True, tokenize=False)
        
        full_text.append(formated_sen)

    return {"text":full_text}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr


    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_folder = "first_results"
    corpus = "lince"

    #HIPERPARAMETERS
    bs = 32
    epochs = 20
    max_seq_length = 1024


    #WANDB
    wandb.init(
    # set the wandb project where this run will be logged
    project="paralel-lince",
    name=f"{model_type}-{lr}-instruct",  # name of the W&B run (optional)
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": model_type,
    "dataset": corpus,
    "epochs": epochs,
    "batch_size":bs,
    },
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    model = AutoModelForCausalLM.from_pretrained(model_chk,quantization_config=bnb_config, device_map="auto")
        
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_chk, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True
    tokenizer.add_eos_token

    if model_chk not in ["Undi95/Meta-Llama-3-8B-hf","Undi95/Meta-Llama-3-8B-Instruct-hf"]:
        tokenizer.add_bos_token, 

    model = prepare_model_for_kbit_training(model)


    peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
            )
    
    model = get_peft_model(model, peft_config)

    dataset = load_parallel(method="base")
    formated_train = dataset["train"].map(train_formatting_function, batched=True)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=True)

    # TRAINING

        
    training_args = SFTConfig(dataset_text_field="text",
            output_dir=save_folder + model_chk + str(lr) + "instruct",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            save_total_limit=2,
            load_best_model_at_end=True,
            num_train_epochs=epochs,
            lr_scheduler_type="inverse_sqrt",
            warmup_ratio=0.1,
            fp16=False,
            optim = "paged_adamw_8bit",
            report_to="wandb",
            run_name=f"{model_chk}-{lr}-instruct",  # name of the W&B run (optional)
            logging_steps=1,
            # remove_unused_columns=False # raises an error otherwise
            #report_to='tensorboard',
        )
    
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            peft_config=peft_config,
            train_dataset=formated_train,
            eval_dataset=formated_dev,
            max_seq_length=max_seq_length,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
            packing=False,
        )

    trainer.train()
    model.config.use_cache = True
    eval_results = trainer.evaluate()