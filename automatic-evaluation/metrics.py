from evaluate import load
import pandas as pd
import argparse
from generate import *
import statistics
import csv

#  FUNCTIONS

def calculate_metrics(p,r):

    bertscore = load("bertscore")
    bleu = load("bleu")
    chrf = load("chrf")

    results_bertscore = bertscore.compute(predictions=p, references=r, model_type="bert-base-multilingual-cased")
    results_bleu = bleu.compute(predictions=p, references=r)
    results_chrf = chrf.compute(predictions=p, references=r)

    return results_bertscore,results_bleu, results_chrf



def truncate(r,p):
    i = 1
    while True:

        ns = " ".join(p.split(".")[:i])

        tokenized_s = nltk.tokenize.word_tokenize(p)
        tokenized_r = nltk.tokenize.word_tokenize(r)
        tokenized_ns = nltk.tokenize.word_tokenize(ns)

        len_r = len(tokenized_r)
        len_ns = len(tokenized_ns)


        if len_r-4 <= len_ns <= len_r+4: #if the length of the generated sentence up to the first punctuation token is appropriate, break
            break
        
        i+=1
        if i>len(p.split(".")):
            ns = " ".join(tokenized_s[:len_r])
            break
    return ns

def get_predictions(ptm,m,s,ins):
    predictions = generate(s,model_path=m,model_chk=ptm,instruct=ins)
    truncated = [truncate(r,p) for p,r in zip(predictions,s)]


    pd.Series(predictions).to_csv(m+"/predictions-dev.tsv",sep="\t")
    pd.Series(truncated).to_csv(m+"/truncated-dev.tsv",sep="\t")

    return predictions , truncated

def process_input(f,m):
    df = pd.read_csv(f,sep="\t")
    df = df
    c = []
    if m == "base":
        for index, row in df.iterrows():
            new_row = "<en>" + str(row["EN"]) + "<cs>" 
            c.append(new_row)
    
    elif m == "len_code":
        for index, row in df.iterrows():
            code = "<"+len_code(str(row["EN"]))+">"
            new_row = "<en>"+ code + str(row["EN"]) + "<cs>" + code
            c.append(new_row)

    elif m == "equal":
        for index, row in df.iterrows():
            new_row = str(row["EN"]) + "=" 
            c.append(new_row)

            
    elif m == "instruct":
        c = list(df["EN"])

    r = list(df["CS"])
    
    return c,r



if __name__ == "__main__":
    gold_dev_path = "/home/mheredia/data/code-switching/LINCE_SPA-EN/dev_drop.tsv"
    gold_test_path = "/home/mheredia/data/code-switching/LINCE_SPA-EN/lince_test.postedited.tsv"

    pred_dev_path = "/home/mheredia/data/code-switching/LINCE_SPA-EN/validation(1).MTen-cs2"
    pred_test_path = "/home/mheredia/data/code-switching/LINCE_SPA-EN/test(1).MTen-cs2"

    gold_dev = pd.read_csv(gold_dev_path, sep="\t",lineterminator="\n",quoting=csv.QUOTE_NONE,quotechar=None)["CS"].tolist()
    gold_test = pd.read_csv(gold_test_path, sep="\t",lineterminator="\n",quoting=csv.QUOTE_NONE,quotechar=None)["CS"].tolist()

    pred_dev = pd.read_csv(pred_dev_path, sep="\t",lineterminator="\n",quoting=csv.QUOTE_NONE)["CS"].tolist()
    pred_test = pd.read_csv(pred_test_path, sep="\t",lineterminator="\n",quoting=csv.QUOTE_NONE)["CS"].tolist()

    truncated_dev = []
    for prediction,reference in zip(pred_dev,gold_dev):
        truncated_dev.append(truncate(reference,prediction))

    
    bsp, blp, bcp = calculate_metrics(truncated_dev,gold_dev)


    print("Raw predictions:")
    print(f'BLEU: {blp["bleu"]}, ChrF: {bcp["score"]}, Bert Score (F1): {statistics.mean(bsp["f1"])}')

    truncated_test=[]
    for prediction,reference in zip(pred_test,gold_test):
        truncated_test.append(truncate(reference,prediction))

    bsp, blp, bcp = calculate_metrics(truncated_test,gold_test)

    print("Raw predictions:")
    print(f'BLEU: {blp["bleu"]}, ChrF: {bcp["score"]}, Bert Score (F1): {statistics.mean(bsp["f1"])}')

    
    # torch.manual_seed(42)
    # np.random.seed(42)

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--folder", type=str)
    # parser.add_argument("--pre_trained", type=str)
    # parser.add_argument("--partition", type=str)
    
    # args = parser.parse_args()


    # folder = args.folder
    # pre_trained = args.pre_trained
    # partition = args.partition

    # if partition == "test":
    #     test = "/home/mheredia/data/code-switching/LINCE_SPA-EN/lince_test.postedited.tsv"

    # if partition == "dev":
    #     test = "/home/mheredia/data/code-switching/LINCE_SPA-EN/dev_drop.tsv"



    # results = {"Model":[],"BLEU":[],"ChrF":[],"BertScore":[],"BLEU_tr":[],"ChrF_tr":[],"BertScore_tr":[]}

    # for m in os.listdir(folder):
    #     for c in os.listdir(folder+"/"+m):
    #         chk = folder+"/"+m+"/"+c
    #         instruct = False

    #         if m.endswith("base"):
    #             method = "base"
            
    #         elif m.endswith("len_code"):
    #             method = "len_code"

    #         elif m.endswith("equal"):
    #             method = "equal"

    #         elif m.endswith("instruct"):
    #             method = "instruct"
    #             instruct = True

    #         concat,references = process_input(test,method)
    #         p, t = get_predictions(pre_trained,chk,concat,instruct)
    #         bsp, blp, bcp = calculate_metrics(p,references)
    #         bst, blt, bct = calculate_metrics(t,references)
            
    #         results["Model"].append(m + " " + c + ":")
    #         results["BLEU"].append(blp["bleu"])
    #         results["ChrF"].append(bcp["score"])
    #         results["BertScore"].append(statistics.mean(bsp["f1"]))
    #         results["BLEU_tr"].append(blt["bleu"])
    #         results["ChrF_tr"].append(bct["score"])
    #         results["BertScore_tr"].append(statistics.mean(bst["f1"]))

    #         print(m + " " + c + ":"+"\n")
    #         print("Raw predictions:")
    #         print(f'BLEU: {blp["bleu"]}, ChrF: {bcp["score"]}, Bert Score (F1): {statistics.mean(bsp["f1"])}')
    #         print("Truncated predictions:")
    #         print(f'BLEU: {blt["bleu"]}, ChrF: {bct["score"]}, Bert Score (F1): {statistics.mean(bst["f1"])}'+"\n"+"\n")

    # df = pd.DataFrame(results)
    # df.to_csv('results-dev.tsv', mode='a', index=False,sep="\t")