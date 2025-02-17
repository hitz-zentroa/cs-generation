import os
import argparse

def parse_conll(file_path, partition):
    with open(file_path, "r", encoding="utf-8") as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if partition == "test":
                if not line:
                    if current_sentence:
                        sentences_test.append(current_sentence)
                        current_sentence = []

                else:
                    parts = line.split("\t")
                    current_sentence.append(tuple(parts))
                
            else:
                if line.startswith("# sent_enum"):
                    if current_sentence:
                        if partition == "train":
                            sentences_train.append(current_sentence)
                        elif partition == "dev":
                            sentences_dev.append(current_sentence)
                        current_sentence = []
                elif line and not line.startswith("#"):
                    parts = line.split("\t")
                    current_sentence.append(tuple(parts))
            
        if current_sentence:
                if partition == "test":
                    sentences_test.append(current_sentence)
                elif partition == "train":
                    sentences_train.append(current_sentence)
                elif partition == "dev":
                    sentences_dev.append(current_sentence)

def process_sentences(sentences):
    output_sentences = []
    
    for sentence in sentences:
        tokens = ""
        tags = []
        

        if len(sentence[0]) == 2:
            for token, tag in sentence:
                tokens += " " + token
                tags.append(tag)
        else:
            for token, tag, _ in sentence:
                tokens += " " + token
                tags.append(tag)
        
        if (tags.count("lang1") >= 2 and tags.count("lang2") >= 2) or (tags.count("eng") >= 2 and tags.count("spa") >= 2):
            output_sentences.append(tokens.strip()) 

    return output_sentences
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rootdir", type=str)
    
    args = parser.parse_args()

    rootdir = args.rootdir # expects a directory with different folders containing the LINCE benchmark 



    sentences_train = []
    sentences_dev = []
    sentences_test = []


    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == "train.conll":
                path_current = os.path.join(subdir, file)
                parse_conll(path_current, "train")
            if file == "dev.conll":
                path_current = os.path.join(subdir, file)
                parse_conll(path_current, "dev")
            if file == "test.conll":
                path_current = os.path.join(subdir, file)
                parse_conll(path_current, "test")


    with open(rootdir+'/lince_train.tsv', 'w') as f:
        for line in process_sentences(sentences_train):
            f.write(line+"\n")

    with open(rootdir+'/lince_dev.tsv', 'w') as f:
        for line in process_sentences(sentences_dev):
            f.write(line+"\n")
    
    with open(rootdir+'/lince_test.tsv', 'w') as f:
        for line in process_sentences(sentences_test):
            f.write(line+"\n")
