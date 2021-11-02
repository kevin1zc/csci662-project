import torch

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'

# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
print(last_layer_features)
