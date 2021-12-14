# csci662-project

This repository is a partial implementation of paper [I like fish, especially dolphins: Addressing Contradictions in 
Dialogue Modeling](https://aclanthology.org/2021.acl-long.134.pdf). Implementations include training models using both 
the utterance-based and the unstructured approaches with RoBERTa as the backbone. Both models are trained with the
DECODE and the ANLI-R3 datasets separately.

1) Download the model and unzip:
```
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.base.tar.gz`
```
2) Format the data: run `format_data.py`
3) Preprocess data: run `preprocess_data.sh`
4) Run training: `train_strucured.sh`, `train_unstructured.sh`, `train_anli.sh`
5) Evaluate the trained models: run `evaluate.py`

Citation of the original paper:
```bibtex
@inproceedings{nie-etal-2021-like,
    title = "{I} like fish, especially dolphins: Addressing Contradictions in Dialogue Modeling",
    author = "Nie, Yixin  and
      Williamson, Mary  and
      Bansal, Mohit  and
      Kiela, Douwe  and
      Weston, Jason",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.134",
    doi = "10.18653/v1/2021.acl-long.134",
    pages = "1699--1713",
    abstract = "To quantify how well natural language understanding models can capture consistency in a general conversation, we introduce the DialoguE COntradiction DEtection task (DECODE) and a new conversational dataset containing both human-human and human-bot contradictory dialogues. We show that: (i) our newly collected dataset is notably more effective at providing supervision for the dialogue contradiction detection task than existing NLI data including those aimed to cover the dialogue domain; (ii) Transformer models that explicitly hinge on utterance structures for dialogue contradiction detection are more robust and generalize well on both analysis and out-of-distribution dialogues than standard (unstructured) Transformers. We also show that our best contradiction detection model correlates well with human judgments and further provide evidence for its usage in both automatically evaluating and improving the consistency of state-of-the-art generative chatbots.",
}
```