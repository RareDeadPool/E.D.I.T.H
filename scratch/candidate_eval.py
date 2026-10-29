import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork
from train import EdithDataset
from torch.utils.data import DataLoader

def eval_candidate():
    device = torch.device('cpu')
    with open('data/metadata_maps.json', 'r', encoding='utf-8') as f:
        maps = json.load(f)
    intents_rev = {v: k for k, v in maps['intents'].items()}

    tokenizer_path = 'models/checkpoints/tokenizer_v8_candidate.json'
    ckpt_path = 'models/checkpoints/edith_ann_v8_candidate.pt'
    if not os.path.exists(ckpt_path):
        print('Candidate checkpoint does not exist yet.')
        return

    tokenizer = EdithTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    model = EdithNeuralNetwork(
        vocab_size=ckpt.get('vocab_size', tokenizer.vocab_size),
        num_intents=len(maps['intents']),
        num_slots=len(maps['slots']),
        num_actions=len(maps['actions']),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=0.0
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    for split in ['test']:
        with open(f'data/{split}_data.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)

        ds = EdithDataset(samples, tokenizer, maps['slots'], max_seq_len=config['model']['max_seq_len'])
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        num_intents = len(maps['intents'])
        tp = [0] * num_intents
        fp = [0] * num_intents
        fn = [0] * num_intents

        with torch.no_grad():
            for batch in loader:
                out = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                preds = out['intent_logits'].argmax(dim=-1).tolist()
                targets = batch['intent'].tolist()
                for p, t in zip(preds, targets):
                    if p == t:
                        tp[t] += 1
                    else:
                        fp[p] += 1
                        fn[t] += 1

        print(f"\n=================== CANDIDATE {split.upper()} INTENT BREAKDOWN ===================")
        print(f"{'Intent':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 70)
        macro_f1 = []
        for i in range(num_intents):
            name = intents_rev[i]
            p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            supp = tp[i] + fn[i]
            macro_f1.append(f1)
            print(f"{name:<22} | {p*100:>9.2f}% | {r*100:>9.2f}% | {f1*100:>9.2f}% | {supp:>8}")
        print("-" * 70)
        mean_f1 = sum(macro_f1)/len(macro_f1)*100
        print(f"Candidate Mean Macro-F1: {mean_f1:.2f}%\n")

if __name__ == '__main__':
    eval_candidate()
