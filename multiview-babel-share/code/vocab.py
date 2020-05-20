import numpy as np
import torch
import torch.utils.data as tud


class Vocab(tud.Dataset):

  def __init__(self, words_to_ids, word_to_subwords, subwords_to_ids):

    loader = tud.DataLoader(self,
                            batch_size=1000,
                            collate_fn=self.collate_fn,
                            num_workers=1)

    self.words_to_ids = words_to_ids
    self.index_to_word = dict(enumerate(words_to_ids))
    self.word_to_subwords = word_to_subwords
    self.subwords_to_ids = subwords_to_ids

    self.loader = loader

  def __getitem__(self, index):  # Note: uses the first pronunciation

    word = self.index_to_word[index]
    seq = self.word_to_subwords[word][0]
    seq_ids = np.array([self.subwords_to_ids[s] for s in seq])

    return {"seq_ids": seq_ids, "word_id": self.words_to_ids[word]}

  def __len__(self):
    return len(self.words_to_ids)

  def collate_fn(self, batch):

    batch_size = len(batch)
    max_seq_len = max([len(ex["seq_ids"]) for ex in batch])
    seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    word_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(batch):
      seq = ex["seq_ids"]
      seqs[i, :len(seq)] = torch.from_numpy(seq)
      seq_lens[i] = len(seq)
      word_ids[i] = torch.tensor(ex["word_id"])

    return {"view2": seqs, "view2_lens": seq_lens, "ids": word_ids}
