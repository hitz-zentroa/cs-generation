from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--input_file", type=str, help="Input file")

    args = parser.parse_args()

    in_path = args.input_file

    model_id = "CohereForAI/c4ai-command-r-v01"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto") 

    output_path = in_path[48:-4]+f".monolingual.tsv"

    initial_batch_size = 32  
    min_batch_size = 4 


    with open(in_path,"r") as dataset:
            lines = dataset.readlines()


    total_lines = len(lines)
    processed_lines = 0
    start_time = time.time()
    batch_size = initial_batch_size


    translated_lines = []
    for line in lines:
 
        prompt = f"""
        Here are five examples of a code-switched text that has been converted to English:
        
        Input: cuando me gusta algo nunca lo hay mi fucking size o no tengo el dinero .
        Output: when I like something there's never my fucking size or I don't have the money .
        
        Input: excelente compartir contigo gracias por tu amistad <user> u rock
        Output: excellent sharing with you thank you for your friendship <user> u rock
        
        Input: fuhk it tacos de frijol
        Output: fuhk it bean tacos
        
        Input: <user> como se llama esa app i wanna play it lmfao
        Output: <user> what's that app called i wanna play it lmfao
        
        Input: i tried putting fake eyelashes on rn lmao me ebarre de glue todo el pinche ojo jajajaja #osoalmil jajaja
        Output: i tried putting fake eyelashes on rn lmao i put glue all over my damn eye hahahaha #superclumsy hahaha
        
        Now convert this code-switched phrase to English. Leave the parts in English as they are, focus on translating the parts in Spanish:
        
        Input:{line}
        Output:
        """    

        messages = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        input_ids = input_ids.to('cuda')

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6, 
        )

        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        translated_lines.append(gen_text)

        processed_lines += 1

        elapsed_time = time.time() - start_time
        avg_time_per_line = elapsed_time / processed_lines
        remaining_lines = total_lines - processed_lines
        eta = remaining_lines * avg_time_per_line
        
        if eta < 60:
            eta_str = f"{eta:.2f} seconds"
        elif eta < 3600:
            eta_str = f"{eta/60:.2f} minutes"
        else:
            eta_str = f"{eta/3600:.2f} hours"

        print(f"Processed {processed_lines}/{total_lines} lines. Estimated time remaining: {eta_str}.")


    with open(output_path,"w") as output:
        for line in translated_lines:
            output.write(line+"\n")