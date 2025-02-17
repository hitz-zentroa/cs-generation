# INITIAL IMPORTS
import numpy as np

import wandb

import torch

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import argparse

from utils.load_parallel import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--method", type=str)
    args = parser.parse_args()

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    method = args.method


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
    name=f"{model_type}-{lr}-{method}",  # name of the W&B run (optional)
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

    if model_chk not in ["Undi95/Meta-Llama-3-8B-hf"]:
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

    dataset = load_parallel(method)

    # TRAINING

        
    training_args = SFTConfig(dataset_text_field="concat",
            output_dir=save_folder + model_chk + str(lr) + method,
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
            run_name=f"{model_chk}-{lr}-{method}",  # name of the W&B run (optional)
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
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            max_seq_length=max_seq_length,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
            packing=False,
        )

    trainer.train()
    model.config.use_cache = True
    eval_results = trainer.evaluate()