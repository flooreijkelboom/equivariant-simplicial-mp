import torch
import torch.nn as nn


class MessageLayer(nn.Module):
    def __init__(self, num_hidden, num_inv):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.edge_inf_mlp = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = scatter_add(messages * edge_weights, index_rec, dim=0)

        return messages_aggr


class UpdateLayer(nn.Module):
    def __init__(self, num_hidden, num_mes):
        super().__init__()
        self.update_mlp = nn.Sequential(
            nn.Linear((num_mes + 1) * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, x, bound_mes, upadj_mes):
        state = x

        if torch.is_tensor(bound_mes):
            state = torch.cat((state, bound_mes), dim=1)

        if torch.is_tensor(upadj_mes):
            state = torch.cat((state, upadj_mes))

        update = self.update_mlp(state)
        return update
