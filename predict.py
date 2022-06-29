import transformers
import torch
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


torch.set_num_threads(2)
MODEL_NAME = 'xlm-roberta-base'
labels_upper = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']
labels_lower = ['av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt',
                'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels_full = labels_upper + labels_lower


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
    ap.add_argument('--labels', choices=['full', 'upper', 'lower'],
                    default='full')
    ap.add_argument('--output', default=None, metavar='FILE',
                    help='Location to save predictions')
    return ap


def predict_labels(tokenizer, model, labels, threshold, text):
    tokenized = tokenizer(text, return_tensors='pt')
    pred = model(**tokenized)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1
    return [labels_full[idx] for idx, prob in enumerate(preds.flatten())
            if prob >= threshold and labels_full[idx] in labels]


def load_models(name_tokenizer, name_fine_tuned):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            name_tokenizer)
    model = torch.load(
            name_fine_tuned, map_location=torch.device('cpu'))
    return tokenizer, model


def main():
    options = argparser().parse_args(sys.argv[1:])
    tokenizer, model = load_models(options.model_name, options.load_model)
    labels = labels_full
    if options.labels == "upper":
        labels = labels_upper
    if options.labels == "lower":
        labels = labels_lower

    with open(options.text, 'r') as infile:
        for line in infile:
            predictions = predict_labels(
                    tokenizer, model, labels, options.threshold, line)
            print(f'{" ".join(predictions)}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
