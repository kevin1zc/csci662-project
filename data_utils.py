from collections import defaultdict
from torch.utils.data import Dataset
import json


class DECODE(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, unstructured=False):
        with open(data_file, 'r') as f:
            data = f.read().splitlines()
            self.data = [json.loads(line) for line in data]
        num_len = set()
        for d in self.data:
            num_len.add(len(d["aggregated_contradiction_indices"]))
        self.unstructured = unstructured

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.parse_json(self.data[idx], self.unstructured)

    # TODO:
    def parse_json(self, instance, unstructured):
        label = instance['aggregated_contradiction_indices']
        is_contradiction = 1 if instance["is_contradiction"] else 0
        all_utterances = []
        # stores utterances as dictionary of lists, keyed by agent_id
        speaker_utterances = defaultdict()
        last_speaker = -1
        for turn in instance['turns']:
            speaker = turn['agent_id']
            # prepend each utterance with special token that denotes the speaker
            all_utterances.append('<{0}> {1}'.format(speaker, turn['text']))
            last_speaker = speaker
            try:
                speaker_utterances[speaker].append(turn['text'])
            except KeyError:
                speaker_utterances[speaker] = [turn['text']]

        if unstructured:
            return all_utterances, is_contradiction
        else:
            utterance_str = []
            for utt in speaker_utterances[last_speaker]:
                utterance_str.append(utt)
            return utterance_str, is_contradiction

    # TODO: This one follows the data preparation described in paper. Need to think of how to make it mini-batches.
    # TODO: Problem: one dialog instance have variable numbers of (prev, last) utterance pairs.
    # def parse_json(self, instance, unstructured):
    #     contradiction_indices = instance['aggregated_contradiction_indices']
    #     is_contradiction = 1 if instance["is_contradiction"] else 0
    #     all_utterances = []
    #     # stores utterances as dictionary of lists, keyed by agent_id
    #     speaker_utterances = defaultdict()
    #     last_speaker = -1
    #     for turn in instance['turns']:
    #         all_utterances.append((turn['agent_id'], turn['text']))
    #         last_speaker = turn['agent_id']
    #         try:
    #             speaker_utterances[turn['agent_id']].append((turn['turn_id'], turn['text']))
    #         except KeyError:
    #             speaker_utterances[turn['agent_id']] = [(turn['turn_id'], turn['text'])]
    #
    #     if unstructured:
    #         return all_utterances, is_contradiction
    #     else:
    #         utterance_pairs = []
    #         utterances_one_side = speaker_utterances[last_speaker]
    #         for i in range(len(utterances_one_side) - 1):
    #             utterance_pairs.append((utterances_one_side[i], utterances_one_side[-1]))
    #         return utterance_pairs, is_contradiction, contradiction_indices
