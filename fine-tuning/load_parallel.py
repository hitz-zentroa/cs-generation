from datasets import Dataset, DatasetDict
import pandas as pd

def load_parallel(directory,format="ds"):

    train_df = pd.read_csv(directory+"train.tsv",sep="\t")
    dev_df = pd.read_csv(directory+"dev.tsv",sep="\t")
    test_df = pd.read_csv(directory+"test.tsv",sep="\t")
    
    dfs = {'train': train_df, 'dev': dev_df, 'test': test_df}

    for name,df in dfs.items():
        concat =  []
        for index, row in df.iterrows():

            if name == "test":
                new_row = str(row["EN"]) + "=" 
            else:
                new_row = str(row["EN"]) + "=" + str(row["CS"])
            concat.append(new_row)
        df["concat"] = concat 

    train_ds = Dataset.from_pandas(train_df, split="train")
    dev_ds = Dataset.from_pandas(dev_df, split="dev")    
    test_ds = Dataset.from_pandas(test_df, split="dev")    

    dataset_dict = DatasetDict({
        'train': train_ds,
        'dev': dev_ds,
        'test': test_ds
    })

    # Return the dataset dictionary or DataFrame based on the requested format
    if format == "ds":
        return dataset_dict

    elif format == "df":
        return train_df,dev_df,test_df
    
    else:
        print("Invalid format.")

