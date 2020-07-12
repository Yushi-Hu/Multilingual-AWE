import torch
import torch.nn.functional as F


class Obj02:

    def __init__(self, margin, k):

        self.margin = margin
        self.k = k

    def __call__(self, x, y, inv):

        n, d = x.shape
        m = y.shape[0]
        k = min(self.k, m - 1)

        if m == 1:
            return torch.tensor(0.0,requires_grad=True).cuda()

        # Compute same-pair similarities
        same = F.cosine_similarity(x, y[inv])

        # Compute all diff-pair similarities
        diff_inv = torch.cat([(inv + i) % m for i in range(1, m)])
        diff = F.cosine_similarity(x.view(n, 1, d), y[diff_inv].view(n, m - 1, d), dim=2).flatten()

        # Find most offending word per utterance: obj0
        diff_word = diff.view(n, m - 1).topk(k, dim=1)[0]
        most_offending_word = F.relu(self.margin + diff_word - same.unsqueeze(-1)).pow(2).mean(dim=1).sqrt()

        # Find most offending utterance per word: obj2
        diff_utt = torch.zeros(m, k, device=diff_word.device)
        for i in range(m):
            diff_utt[i] = diff[diff_inv == i].topk(k)[0]
        most_offending_utt = F.relu(self.margin + diff_utt[inv] - same.unsqueeze(-1)).pow(2).mean(dim=1).sqrt()

        return most_offending_word.sum() + most_offending_utt.sum()

