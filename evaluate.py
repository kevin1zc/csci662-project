import torch
import json
from fairseq.models.roberta import RobertaModel
from collections import defaultdict
from tqdm import tqdm


def evaluate_structured(roberta):
    print("Evaluating Utterance-based model on DECODE dataset:")
    data_file = "decode_v0.1/test.jsonl"

    with open(data_file, 'r') as f:
        raw_data = f.read().splitlines()
        raw_data = [json.loads(line) for line in raw_data]

    tp = tn = fp = fn = 0
    instance_correct = 0
    instance_strict_correct = 0
    for instance in tqdm(raw_data):
        contradiction_idx = instance['aggregated_contradiction_indices']
        instance_label = 1 if instance["is_contradiction"] else 0

        speaker_utterances = defaultdict(list)
        last_speaker = -1
        for turn in instance['turns']:
            speaker = turn['agent_id']
            last_speaker = speaker
            speaker_utterances[speaker].append(turn['text'])
        utterances = speaker_utterances[last_speaker]

        instance_label_pred = 0
        all_pairs_correct = True
        for i in range(len(utterances) - 1):
            turn_idx = i * 2 + last_speaker
            pair_label = 1 if turn_idx in contradiction_idx else 0

            tokens = roberta.encode(utterances[i], utterances[-1])
            pair_label_pred = roberta.predict('decode_head', tokens).argmax()
            if pair_label_pred == pair_label == 1:
                instance_label_pred = 1
                tp += 1
            elif pair_label_pred == pair_label == 0:
                tn += 1
            elif pair_label_pred == 1 and pair_label == 0:
                instance_label_pred = 1
                fp += 1
                all_pairs_correct = False
            else:  # label_pred==0, label==1
                fn += 1
                all_pairs_correct = False
        if all_pairs_correct:
            instance_strict_correct += 1
        if instance_label_pred == instance_label:
            instance_correct += 1
    print(f"  MT: {instance_correct / len(raw_data)}")
    print(f"  MT strict: {instance_strict_correct / len(raw_data)}")
    print(f"  SE F1: {tp / (tp + 0.5 * (fp + fn))}")


def evaluate_unstructured(roberta):
    print("Evaluating Unstructured model on DECODE dataset:")
    data_file = "decode_v0.1/test.jsonl"

    with open(data_file, 'r') as f:
        raw_data = f.read().splitlines()
        raw_data = [json.loads(line) for line in raw_data]

    instance_correct = 0
    unevaluated = 0
    for instance in tqdm(raw_data):
        all_utterances = []
        label = 1 if instance["is_contradiction"] else 0
        for turn in instance['turns']:
            speaker = turn['agent_id']
            # prepend each utterance with special token that denotes the speaker
            all_utterances.append('<{0}> {1}'.format(speaker, turn['text']))
        prev_utterances = ' '.join(all_utterances[:-1])
        tokens = roberta.encode(prev_utterances, all_utterances[-1])
        try:
            label_pred = roberta.predict('decode_head', tokens).argmax()
            if label_pred == label:
                instance_correct += 1
        except:
            unevaluated += 1

    print(f"  MT: {instance_correct / len(raw_data)}")
    print(f"  Unevaluated: {unevaluated}")


if __name__ == "__main__":
    # roberta = RobertaModel.from_pretrained(
    #     'model_structured/checkpoints',
    #     checkpoint_file='checkpoint_best.pt',
    #     data_name_or_path='decode-bin/structured'
    # )
    # roberta.eval().cuda()
    # evaluate_structured(roberta)

    roberta = RobertaModel.from_pretrained(
        'model_unstructured/checkpoints',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='decode-bin/unstructured'
    )
    roberta.eval().cuda()
    evaluate_unstructured(roberta)
