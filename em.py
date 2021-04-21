import argparse
import os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read(path):
    word_map = {}  # id to word
    with open(os.path.join(path, "20news.vocab"), "r") as f:
        flines=f.readlines()
        W = len(flines)
        freq_map = torch.zeros([W])
        for line in flines:
            l = line.strip().split()
            i = int(l[0])
            word = l[1]
            f = int(l[2])
            word_map[i] = word
            freq_map[i] = f

    with open(os.path.join(path, "20news.libsvm"), "r") as f:
        flines=f.readlines()
        D = len(flines)
        T = torch.zeros([D,W])
        for i, line in enumerate(flines):
            l = line.strip().split()
            for pair in l[1:]:
                l2 = pair.split(":")
                word, freq = int(l2[0]), int(l2[1])
                T[i][word] = freq
    
    return D, W , word_map, T


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="20news",
                    help="Data directory")
parser.add_argument("--K", type=int, default=10,
                    help="Data directory")
args = parser.parse_args()

D,W,word_map, T = read(args.data)

T=T.to(device)

for K in [10, 20, 30, 50]:
    pi = torch.softmax(torch.randn([K]),dim=0).to(device)
    mu = torch.softmax(torch.randn([W,K]),dim=1).to(device)
    step = 0
    eps = 1e-10
    pi_old = torch.zeros([K]).to(device)

    while torch.norm(pi_old - pi) > 1e-3:
        # E step
        gamma = torch.softmax(  T.mm(torch.log(mu+eps)) + torch.log(pi).t() ,dim =1 )

        # M step
        pi_old = pi
        pi = torch.mean(gamma,dim=0)
        mu = T.t().mm(gamma)
        mu = mu/torch.sum(mu,dim=0)
        
        print("K=%d, step=%d, onestep-diff=%f" % (K, step, torch.norm(pi_old - pi)))
        step += 1

    for k in range(K):
        print("Topic %d:" % k,end='')
        topics = np.argsort(mu.cpu().numpy()[:,k])[::-1]
        for i in range(min(10, len(topics))):
            print(" %s"% (word_map[topics[i]]),end=',')
        print("\n")