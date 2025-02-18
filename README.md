# Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data
Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. We present a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license.

Link to the paper: [URL]

## Requirements
`requirements.txt` includes the list of dependencies.

To install all:

```
pip install -r requirements.txt
```

## Parallel data creation
The directory `parallel-data-creation` contains the scripts to generate the parallel corpus CS - English.
The scripts assume a directory with the original files of LINCE. The corpus will be saved in that same directory. To download the original LINCE dataset: [LINCE Benchmark](https://ritual.uh.edu/lince/).

Example of usage:

```
bash parallel-data-creation/parallel-data-creation.sh "path-to-lince-directory"
```

## Download dataset
The parallel corpus *EN2CS* can be downloaded here: [https://ixa2.si.ehu.eus/mheredia/EN2CS.zip](https://ixa2.si.ehu.eus/mheredia/EN2CS.zip)

The final dataset includes changes beyond the scope of the scripts, as a subset of the test set has been post-edited and with that subset, the splits have been deduplicated. Consequently, the following sections assume a directory that contains the *EN2CS* dataset as provided.

## Fine-tuning
The directory `fine-tuning` contains the scripts to fine-tune the different models on the task of CS generation.


Example usage to train a base model:
```
python fine-tuning/train_decoder.py --model "mistralai/Mistral-7B-v0.3" --model_type "mistral" --lr 0.0005
```

And to train an instruct model:
```
python fine-tuning/train_instruct.py --dataset_path "path-to-en2cs-directory/" --save_path "save_path" --model "mistralai/Mistral-7B-Instruct-v0.3" --model_type "mistral" --lr $lr
```

To replicate all the experiments included in the paper, run:
```
bash fine-tuning/all-models.sh "path-to-en2cs-directory/" "save_path"
```

## Qualitative Evaluation
The directory `qualitative-evaluation` contains the scripts to obtain scores and figures from the manual evaluation.

## Automatic Evaluation
The directory `automatic-evaluation` contains the scripts to calculate the automatic metrics and the correlation with human evaluation.

## Citation
The paper that explains the dataset and experiments can be cited as follows:

```
@inproceedings{
}
```

# License