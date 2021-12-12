import os
import random
from collections import defaultdict
import json


def format_structured(datadir, split, raw_data):
    samples = []
    samples_neg = []
    for instance in raw_data:
        contradiction_idx = instance['aggregated_contradiction_indices']
        speaker_utterances = defaultdict(list)
        last_speaker = -1
        for turn in instance['turns']:
            speaker = turn['agent_id']
            last_speaker = speaker
            speaker_utterances[speaker].append(turn['text'])
        utterances = speaker_utterances[last_speaker]
        for i in range(len(utterances) - 1):
            turn_idx = i * 2 + last_speaker
            label = 1 if turn_idx in contradiction_idx else 0
            if label == 1:
                samples.append((utterances[i], utterances[-1], label))
            else:
                samples_neg.append((utterances[i], utterances[-1], label))
    print(f"Number of positive/negative utterance pairs in {split} set: {len(samples)}")
    random.shuffle(samples_neg)
    samples_neg = samples_neg[:len(samples)]
    samples.extend(samples_neg)
    random.shuffle(samples)
    out_fname = split
    save_data(samples, os.path.join(datadir, "formatted", "structured"), out_fname)


def format_unstructured(datadir, split, raw_data):
    samples = []
    for instance in raw_data:
        all_utterances = []
        label = 1 if instance["is_contradiction"] else 0
        for turn in instance['turns']:
            speaker = turn['agent_id']
            # prepend each utterance with special token that denotes the speaker
            all_utterances.append('<{0}> {1}'.format(speaker, turn['text']))
        prev_utterances = ' '.join(all_utterances[:-1])
        samples.append((prev_utterances, all_utterances[-1], label))
    random.shuffle(samples)

    out_fname = split
    save_data(samples, os.path.join(datadir, "formatted", "unstructured"), out_fname)


def save_data(samples, save_dir, out_fname):
    os.makedirs(save_dir, exist_ok=True)
    f1 = open(os.path.join(save_dir, out_fname + '.input0'), 'w')
    f2 = open(os.path.join(save_dir, out_fname + '.input1'), 'w')
    f3 = open(os.path.join(save_dir, out_fname + '.label'), 'w')
    for sample in samples:
        f1.write(sample[0] + '\n')
        f2.write(sample[1] + '\n')
        f3.write(str(sample[2]) + '\n')
    f1.close()
    f2.close()
    f3.close()


def format_decode():
    datadir = 'decode_v0.1'
    for split in ['train', 'dev']:
        data_file = os.path.join(datadir, split + '.jsonl')

        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
            raw_data = [json.loads(line) for line in raw_data]

        format_structured(datadir, split, raw_data)
        format_unstructured(datadir, split, raw_data)


def format_anli():
    datadir = 'anli_v1.0/R3'
    for split in ['train', 'dev']:
        data_file = os.path.join(datadir, split + '.jsonl')
        samples = []
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
            raw_data = [json.loads(line) for line in raw_data]
        for instance in raw_data:
            label = 1 if instance['label'] == 'c' else 0
            samples.append((instance['context'], instance['hypothesis'], label))
        random.shuffle(samples)
        out_fname = split
        save_data(samples, os.path.join(datadir, "formatted"), out_fname)


if __name__ == '__main__':
    format_decode()
    format_anli()
