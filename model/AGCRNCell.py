import torch
import torch.nn as nn
from model.AGCN import AVWGCN

class AGCRNCell(nn.Module):
    def __init__(self, model_name, p1, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(model_name, p1, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim) #why 2*dim_out? z_t, r_t
        self.update = AVWGCN(model_name, p1, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1) # z,r exchange
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        
        #print(x.shape, state.shape, input_and_state.shape)
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)