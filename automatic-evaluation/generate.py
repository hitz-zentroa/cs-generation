from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate (sentences, model_path, model_chk, max_new_tokens = 50,instruct=False):
    tokenizer = AutoTokenizer.from_pretrained(model_chk,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_chk, device_map="auto")
    peft_model_id = model_path
    model = PeftModel.from_pretrained(model, peft_model_id)
    if instruct:
        full_text = []
        for s in sentences:
                formated_sen_chat = [
                {"role": "system", "content": "You are a bilingual speaker of English and Spanish. Translate the following English sentence into code-switched text between both languages:"},
                {"role": "user", "content":s},
                ]

                formated_sen = tokenizer.apply_chat_template(formated_sen_chat, add_generation_prompt=True, tokenize=False)
                
                full_text.append(formated_sen)
    
        sentences = full_text.copy()
    batch_size = 500 
    outputs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        gen_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens,do_sample=True)        
        
        for i, input_sentence in enumerate(batch):
                input_length = inputs.input_ids.shape[-1]  # Adjust if you need to handle different sentence lengths
                response_tokens = gen_tokens[i][input_length:]  # Remove the input part from the generated tokens
                generated_sentence = tokenizer.decode(response_tokens, skip_special_tokens=True)
                outputs.append(generated_sentence.strip().split("\n")[0].strip())
    
    return(outputs)

