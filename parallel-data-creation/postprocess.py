import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str, help="Input file")

args = parser.parse_args()

input_path = args.input_file

def clean(df_dirty):

    list_cleaned = []
    words = r'fucking|output|translation|traducción|traduccion|translated|traducido|traducida|codeswitch[ing|ed|]|span|la gatita|zomber'
    for index, row in df_dirty.iterrows():
        if re.search(words, row["EN"].lower()):
            continue
        if re.search(words, row["ES"].lower()):
            continue    
        list_cleaned.append(row)
    cleaned = pd.DataFrame(list_cleaned)

    return cleaned


df = pd.read_csv(input_path, sep ="\t")
df = df.drop_duplicates()
clean(df).to_csv(input_path[:-4]+".post.tsv",sep="\t",index=False)