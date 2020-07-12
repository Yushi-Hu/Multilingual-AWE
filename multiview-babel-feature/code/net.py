import logging as log
import torch

import layers.rnn
import layers.linear
import utils.saver


class MultiViewRNN(utils.saver.NetSaver):

    def __init__(self, config, feat_dim, num_subwords,
                 loss_fun=None, use_gpu=False):
        super(MultiViewRNN, self).__init__()

        self["view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                               inputs=feat_dim,
                                               hidden=config.view1_hidden,
                                               bidir=config.view1_bidir,
                                               dropout=config.view1_dropout,
                                               layers=config.view1_layers)

        self["view2"] = layers.rnn.RNN_default(cell=config.view2_cell,
                                               num_embeddings=num_subwords,
                                               inputs=config.view2_inputs,
                                               hidden=config.view2_hidden,
                                               bidir=config.view2_bidir,
                                               dropout=config.view2_dropout,
                                               layers=config.view2_layers)

        log.info(f"view1: feat_dim={feat_dim}")
        log.info(f"view2: num_subwords={num_subwords}")

        if config.projection is not None:
            self["proj"] = layers.linear.Linear(in_features=self["view1"].d_out,
                                                out_features=config.projection)

        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward_view(self, view, batch, numpy=False):
        _, _, out = self[view](batch[view], batch[f"{view}_lens"])
        if "proj" in self:
            out = self["proj"](out)
        if numpy:
            out = out.cpu().numpy()
        return out

    def forward(self, batch, numpy=False):
        view1_out = self.forward_view("view1", batch, numpy=numpy)
        view2_out = self.forward_view("view2", batch, numpy=numpy)
        return view1_out, view2_out

    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)


class MultiViewRNN_Phonetic(utils.saver.NetSaver):

    def __init__(self, config, feat_dim, phone_feat_dim,
                 loss_fun=None, use_gpu=False):
        super().__init__()

        self["view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                               inputs=feat_dim,
                                               hidden=config.view1_hidden,
                                               bidir=config.view1_bidir,
                                               dropout=config.view1_dropout,
                                               layers=config.view1_layers)

        self["oh2emb"] = layers.linear.Linear(in_features=phone_feat_dim,
                                              out_features=config.view2_inputs,
                                              bias=False)

        self["view2"] = layers.rnn.RNN_default(cell=config.view2_cell,
                                               inputs=config.view2_inputs,
                                               hidden=config.view2_hidden,
                                               bidir=config.view2_bidir,
                                               dropout=config.view2_dropout,
                                               layers=config.view2_layers)

        log.info(f"view1: feat_dim={feat_dim}")
        log.info(f"view2: phone_feat_dim={phone_feat_dim}")

        if config.projection is not None:
            self["proj"] = layers.linear.Linear(in_features=self["view1"].d_out,
                                                out_features=config.projection)

        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward_view1(self, view, batch, numpy=False):
        _, _, out = self[view](batch[view], batch[f"{view}_lens"])
        if "proj" in self:
            out = self["proj"](out)
        if numpy:
            out = out.cpu().numpy()
        return out

    def forward_view2(self, view, batch, numpy=False):
        embeded = self["oh2emb"](batch[view])
        _, _, out = self[view](embeded, batch[f"{view}_lens"])
        if "proj" in self:
            out = self["proj"](out)
        if numpy:
            out = out.cpu().numpy()
        return out

    def forward(self, batch, numpy=False):
        view1_out = self.forward_view1("view1", batch, numpy=numpy)
        view2_out = self.forward_view2("view2", batch, numpy=numpy)
        return view1_out, view2_out

    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)