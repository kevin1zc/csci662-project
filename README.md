# csci662-project

1) Download necessary files: 
```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'```
```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'```
```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  ```

2) Format the data:
   Run ```format_data.py```

4) Preprocess data: run ```preprocess_data.sh```
5) Run training: ```train_strucured.sh```, ```train_unstructured.sh```