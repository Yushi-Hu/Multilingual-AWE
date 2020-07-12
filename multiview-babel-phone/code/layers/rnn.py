import torch

import utils.saver


class RNN_default(torch.nn.Module, utils.saver.Saver):

  def __init__(self, cell, inputs, hidden, bidir, dropout=0.0, layers=1,
               num_embeddings=None):
    super(RNN_default, self).__init__()

    if num_embeddings is not None:
      self.emb = torch.nn.Embedding(num_embeddings=num_embeddings + 1,
                                    embedding_dim=inputs,
                                    padding_idx=0)

    self.rnn = getattr(torch.nn, cell)(input_size=inputs,
                                       hidden_size=hidden,
                                       bidirectional=bidir,
                                       num_layers=layers,
                                       dropout=dropout if layers > 1 else 0.,
                                       batch_first=True)

    self.d_out = 2 * hidden if bidir else hidden

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, x, lens):

    x = x.to(self.device)

    if hasattr(self, "emb"):
      x = self.emb(x)

    x = torch.nn.utils.rnn.pack_padded_sequence(
        x, lens, batch_first=True, enforce_sorted=False)

    x, h = self.rnn(x)

    x, lens = torch.nn.utils.rnn.pad_packed_sequence(
        x, batch_first=True)

    if isinstance(h, tuple):
      h = h[0]

    if self.rnn.bidirectional:
      h = torch.cat((h[-2], h[-1]), dim=1)
    else:
      h = h[-1]

    return x, lens, h
