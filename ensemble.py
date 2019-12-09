
import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


parser = ArgumentParser()

parser.add_argument("model_a_logits", type=str,
                    help="Path to the first model's logits.")

parser.add_argument("model_b_logits", type=str,
                    help="Path to the second model's logits.")

parser.add_argument("save_fname", type=str,
                    help="Name to give final results file.")

args = parser.parse_args()


def combine(aFile, bFile):
    a = np.load(aFile)
    b = np.load(bFile)

    idx = min(a.shape[0], b.shape[0])

    c = a[:idx] + b[:idx]
    return c.argmax(axis=1).tolist()


def main():
    identifiers = ['_'.join(x.split('_')[:-1]) for x in os.listdir(args.model_a_logits)]
    identifiers = list(set(identifiers))

    dirname_a = os.path.dirname(args.model_a_logits)
    dirname_b = os.path.dirname(args.model_b_logits)

    result = {}

    for id_ in tqdm(identifiers):
        verb_fname = id_ + "_verb.npy"
        noun_fname = id_ + "_noun.npy"

        verb = combine(os.path.join(dirname_a, verb_fname),
                       os.path.join(dirname_b, verb_fname))

        noun = combine(os.path.join(dirname_a, noun_fname),
                       os.path.join(dirname_b, noun_fname))

        result[id_] = list(zip(verb, noun))

    pickle.dump(result, open(os.path.join("./output/", args.save_fname), "wb"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
