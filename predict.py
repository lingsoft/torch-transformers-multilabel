import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import transformers
import torch


torch.set_num_threads(2)
MODEL_NAME = 'xlm-roberta-base'
labels_general = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']
labels_sub = ['av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt',
              'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels_full = labels_general + labels_sub


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--text', metavar='FILE', required=True,
                    help='Text to be predicted')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='Load model from file')
    ap.add_argument('--threshold', default=0.4, metavar='FLOAT', type=float,
                    help='threshold for calculating f-score')
    ap.add_argument('--labels', choices=['full', 'general'],
                    default='full')
    ap.add_argument('--output', default=None, metavar='FILE',
                    help='Location to save predictions')
    return ap


def load_models(name_tokenizer, name_fine_tuned):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            name_tokenizer)
    model = torch.load(
            name_fine_tuned, map_location=torch.device('cpu'))
    return tokenizer, model


def predict(tokenizer, model, text):
    tokenized = tokenizer(text, return_tensors='pt')
    pred = model(**tokenized)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
    return [(idx, prob) for idx, prob in enumerate(
        probs.detach().cpu().numpy()[0])]


def print_labels(probs, labels, threshold):
    probs.sort(key=lambda x: x[1], reverse=True)
    for idx, prob in probs:
        label = labels_full[idx]
        if prob > threshold and label in labels:
            print((label, prob), end=" ")
    print()


def main():
    options = argparser().parse_args(sys.argv[1:])
    tokenizer, model = load_models(options.model_name, options.load_model)
    labels = labels_full
    if options.labels == "general":
        labels = labels_general

    with open(options.text, 'r') as infile:
        for line in infile:
            probs = predict(tokenizer, model, line)
            print_labels(probs, labels, options.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
