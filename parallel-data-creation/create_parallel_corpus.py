# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import re

class DocumentProcessor:
    def __init__(self, tag):
        self.tag = tag

    def process_lines(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        processed_lines = []
        
        for line in lines:
            if line.startswith(self.tag):
                processed_lines.append(line)
            else:
                if processed_lines:
                    processed_lines[-1] = processed_lines[-1].strip() + " " + line
                else:
                    processed_lines.append(line.strip())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)


def get_answer(text):
        pattern = r'(?<=<\|CHATBOT_TOKEN\|>)[^\n]+'  # Positive lookbehind for the tag
        match = re.search(pattern, text)
        if match:
            return match.group().strip()
        else:
            return "PROBLEM WITH THIS LINE"


parser = argparse.ArgumentParser()

parser.add_argument("--cs_path", type=str)
parser.add_argument("--en_path", type=str, required=False)

args = parser.parse_args()

cs_path = args.cs_path
en_path = args.en_path


df = pd.DataFrame()

# Initialize the DocumentProcessor with the tag you're interested in
tag = '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>'  # Specify the tag to be used for filtering lines
processor = DocumentProcessor(tag)

processed_en_path = en_path[:-4] + "_processed.txt"
processor.process_lines(en_path, processed_en_path)
en_path = processed_en_path


with open(cs_path, "r") as cs_lines:
    cs_list = cs_lines.read().split("\n")

with open(en_path, "r") as en_lines:
    en_list_current = en_lines.read().split("\n")
en_list = list(map(get_answer, en_list_current))



df["CS"] = cs_list
df["EN"] = en_list

df.to_csv(cs_path[:-4] + "_parallel.tsv", sep="\t", index=None)
