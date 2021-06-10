# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# EDITED in 10/6/2021 by Nicola Sartorato

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import pandas

parser = argparse.ArgumentParser(description='Find probability distribution for the next word for each word in the corpus')

parser.add_argument('--data', type=str,
                    help='location of the data corpus for LM testing')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--corpus', type=str, 
                    help='path for the text file with sentences on rows separated by spaces and ended with <eos>')
parser.add_argument('--output', type=str, 
                    help='path for the output file in which probs have to be saved')

args = parser.parse_args()


def get_prob_distributions(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    hidden = model.init_hidden(eval_batch_size)

    ret = []
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, _ = get_batch(data_source, i, seq_len)
            
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, len(dictionary))

            idx_words = [i[0] for i in data.numpy()]
            probs = F.softmax(output_flat, dim=1)
            probs_np = probs.cpu().numpy()
            
            l = []
            
            for i in range(len(idx_words)):
                if i < len(idx_words) - 1:
                    l.append({
                        "word": str(dictionary.idx2word[idx_words[i]]),
                        "next_word": str(dictionary.idx2word[idx_words[i+1]]),
                        "P_next_word": probs_np[i][idx_words[i+1]],
                        "P_distr_word": list(probs_np[i])
                    })
                else:
                    l.append({
                        "word": str(dictionary.idx2word[idx_words[i]]),
                        "next_word": None,
                        "P_next_word": None,
                        "P_distr_word": list(probs_np[i])
                    })

            hidden = repackage_hidden(hidden)
            ret += l

    return ret


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    print("Loading the model")
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = 1
seq_len = 20
dictionary = dictionary_corpus.Dictionary(args.data)

print("Computing probabilities")
test_data = batchify(dictionary_corpus.tokenize(dictionary, args.corpus), eval_batch_size, args.cuda)
l = get_prob_distributions(test_data)
df = pandas.DataFrame(l)
df.to_csv(args.output, index=None)
print("Probabilities saved to", args.output)
