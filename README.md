# Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data
Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. We present a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license.

## Download dataset
The parallel corpus *EN2CS* can be downloaded here: [https://ixa2.si.ehu.eus/mheredia/EN2CS.zip](https://ixa2.si.ehu.eus/mheredia/EN2CS.zip)

## Parallel data creation
The directory `parallel-data-creation` contains the scripts to generate the parallel corpus CS - English.
The scripts assume a directory with the original files of LINCE. To download the original LINCE dataset: [LINCE Benchmark](https://ritual.uh.edu/lince/).


## Fine-tuning
The directory `fine-tuning` contains the scripts to fine-tune the different models on the task of CS generation.


## Qualitative Evaluation
The directory `qualitative-evaluation` contains the scripts to obtain scores and figures from the manual evaluation.

## Automatic Evaluation
The directory `automatic-evaluation`  contains the scripts to calculate the automatic metrics and the correlation with human evaluation.

## Citation
The paper that explains the dataset and experiments can be cited as follows:

```
@inproceedings{
}
```

