# INITIAL IMPORTS
import numpy as np
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import argparse
from load_parallel import *



def train_formatting_function(data):
    """
    Template for training
    """
    full_text = []
    
    for i in range(len(data['CS'])):

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

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    model_chk = args.model
    model_type = args.model_type
    lr = args.lr
    save_path = args.save_path
    dataset_path = args.dataset_path


    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    

    #HIPERPARAMETERS
    bs = 32
    epochs = 20
    max_seq_length = 1024



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

    dataset = load_parallel(dataset_path)
    formated_train = dataset["train"].map(train_formatting_function, batched=True)
    formated_dev = dataset["dev"].map(train_formatting_function, batched=True)

    # TRAINING

        
    training_args = SFTConfig(dataset_text_field="text",
            output_dir=save_path + model_chk + str(lr)+ "instruct",
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
            logging_steps=1
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