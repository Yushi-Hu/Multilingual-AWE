import torch

import utils.saver


class CNN_1D(torch.nn.Module, utils.saver.Saver):

  def __init__(self, inputs, channels, kernel_size, stride, dilation,
               padding=0, bias=True, padding_mode="zeros", num_embeddings=None):
    super(CNN_1D, self).__init__()

    if num_embeddings is not None:
      self.emb = torch.nn.Embedding(num_embeddings=num_embeddings + 1,
                                    embedding_dim=inputs,
                                    padding_idx=0)

    self.cnn = torch.nn.Conv1d(in_channels=inputs,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=bias)

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, x, lens):

    x = x.to(self.device)

    if hasattr(self, "emb"):
      x = self.emb(x)

    x = self.cnn(x.transpose(1, 2)).transpose(1, 2)

    lens = ((lens - self.cnn.kernel_size).float() / self.cnn.stride).long() + 1

    return x.contiguous(), lens
