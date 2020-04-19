import argparse
import os
import numpy as np


def read(path):
    word_map = {}  # id to word
    freq_map = {}
    with open(os.path.join(path, "20news.vocab"), "r") as f:
        for line in f:
            l = line.strip().split()
            i = int(l[0])
            word = l[1]
            f = int(l[2])
            word_map[i] = word
            freq_map[i] = f

    documents = []
    word_to_doc = [[] for _ in range(len(word_map))]
    with open(os.path.join(path, "20news.libsvm"), "r") as f:
        for i, line in enumerate(f):
            doc = {}
            l = line.strip().split()
            for pair in l[1:]:
                l2 = pair.split(":")
                word, freq = int(l2[0]), int(l2[1])
                doc[word] = freq
                word_to_doc[word].append(i)
            documents.append(doc)
    
    return word_map, freq_map, documents, word_to_doc


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="20news",
                    help="Data directory")
parser.add_argument("--K", type=int, default=10,
                    help="Data directory")
args = parser.parse_args()

word_map, freq_map, documents, word_to_doc = read(args.data)
W = len(word_map)
D = len(documents)
K = args.K
# for K in [10, 20, 50, 100]:

# initialization
pi = np.random.random([K])
mu = np.random.random([K, W])
for k in range(K):
    pi[k] /= np.sum(pi[k])
for k in range(K):
    mu[k] /= np.sum(mu[k])

step = 0
eps = 1e-10
pi_old = np.zeros([K], dtype=float)
while np.linalg.norm(pi_old - pi) > 1e-3:
    # E step
    gamma = np.zeros([D, K], dtype=float)
    for d in range(D):
        for k in range(K):
            gamma[d][k] = np.log(pi[k])
            for w in documents[d]:
                gamma[d][k] += documents[d][w] * np.log(mu[k][w] + eps)
        maxn = max(gamma[d])
        for k in range(K):
            gamma[d][k] = np.exp(gamma[d][k] - maxn)
        gamma[d] = gamma[d] / np.sum(gamma[d])

    # M step
    pi_old = pi
    pi = np.sum(gamma, axis=0) / np.sum(gamma)
    for k in range(K):
        for w in range(W):
            mu[k][w] = 0
            for d in word_to_doc[w]:
                mu[k][w] += gamma[d][k] * documents[d][w]
        mu[k] = mu[k] / np.sum(mu[k])
    
    print("K=%d, step=%d, norm-diff=%f" % (K, step, np.linalg.norm(pi_old - pi)))
    step += 1

for k in range(K):
    print("Topic %d:" % k)
    topics = np.argsort(mu[k])[::-1]
    for i in range(min(10, len(topics))):
        print("    %s: %f" % (word_map[topics[i]], mu[k][topics[i]]))
