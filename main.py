import torch
import numpy as np
import json

from collections import defaultdict

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'

# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
print(last_layer_features)

# TODO optimize for CUDA on CARC?
def batch_generator(data, batch_size, unstructured=True, isLabelled=True):
	n = len(data)
	# creates random permuation of indices [0,...,n-1]
	permutedIndices = np.random.permutation(range(n))

	for k in range(0, n, batch_size):
		batch = [parse_json(data[i], unstructured) for i in permutedIndices[k:k+batch_size]]
		# batch = torch.tensor(np.array(batch))

		yield batch

		# TODO integrate support for both labelled and unlabelled datasets
		# batch_X = torch.tensor(batch_X) if tensorize else batch_X
		# if isLabelled:
		# 	batch_labels = np.array([self.one_hot_labels[i] for i in permutedIndices[k:k+batch_size]])
		# 	batch_labels = torch.tensor(batch_labels) if tensorize else batch_labels

		# 	yield batch_X, batch_labels			
		# else:
		# 	yield batch_X

# parses one instance of json-structured data
# if unstructured, it will embed and concatenate all utterances
# else, it will just embed the utterances from the final speaker
def parse_json(instance, unstructured):
	label = instance['aggregated_contradiction_indices']
	all_utterances = ''
	# stores utterances as dictionary of lists, keyed by agent_id
	speaker_utterances = defaultdict()
	last_speaker = -1
	for turn in instance['turns']:
		all_utterances += turn['text']
		last_speaker = turn['agent_id']
		try:
			speaker_utterances[turn['agent_id']].append(turn['text'])
		except KeyError:
			speaker_utterances[turn['agent_id']] = [turn['text']]

	if unstructured:
		return all_utterances, label
	else:
		utterance_str = ''
		for utt in speaker_utterances[last_speaker]:
			utterance_str += utt
		return utterance_str, label

# loads jsonl file into memory (hopefully this won't crash on CARC :P)
def load_jsonl(fname):
	data = None
	with open(fname, 'r') as f:
		data = f.read().splitlines()
		data = [json.loads(line) for line in data]

	return data
