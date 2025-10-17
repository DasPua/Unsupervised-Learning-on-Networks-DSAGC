import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    def __init__(
        self,
        input_features_dimension,
        output_features_dimension,
        dropout,
        alpha,
        concat=True,
    ):
        super(GraphAttention, self).__init__()
        self.input_features_dimension = input_features_dimension
        self.output_features_dimension = output_features_dimension
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(
            torch.empty(size=(input_features_dimension, output_features_dimension))
        )
        self.a = nn.Parameter(torch.empty(size=(2 * output_features_dimension, 1)))
        self.leakyReLU = nn.LeakyReLU(self.alpha)

        nn.init.xavier_uniform(self.W.data, gain=1.414)
        nn.init.xavier_uniform(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[: self.output_features_dimension, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_features_dimension :, :])
        ##here what we are doing is that we have to concatenate the Whi and Whj and then calculate the leaky relu
        ##but we can do that using the a which has the dimension twice of Wh so we can just matmul and then add before leakyrelu
        e = Wh1 + Wh2.T
        return self.leakyReLU(e)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism(Wh)

        zerovec = -9e15 * torch.ones_like(
            e
        )  ##this returns a ones tensor just like the size of the input
        attention = torch.where(
            adj > 0, e, zerovec
        )  ##this will take condition , input, otherwise
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_next = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(
                h_next
            )  ##this is exponential linear unit for removing the negative parts
        else:
            return h_next

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.input_features_dimension)
            + " -> "
            + self(self.output_features_dimension)
            + ")"
        )


class MultiheadedGraphAttention(nn.Module):
    def __init__(
        self,
        number_of_features,
        number_of_hidden_layers,
        number_of_output_classes,
        number_of_heads,
        dropout,
        alpha,
    ):
        super(MultiheadedGraphAttention, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttention(
                number_of_features,
                number_of_hidden_layers,
                dropout=dropout,
                alpha=alpha,
                concat=True,
            )
            for _ in range(number_of_heads)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

            self.output_attention = GraphAttention(
                number_of_hidden_layers * number_of_heads,
                number_of_output_classes,
                dropout=dropout,
                alpha=alpha,
                concat=False,
            )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attention(x, adj) for attention in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.output_attention(x, adj))
        return F.log_softmax(x, dim=1)
