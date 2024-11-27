import torch
from torch import nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=feature_dim * 2)
        self.fc2 = nn.Linear(in_features=feature_dim * 2, out_features=feature_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, input_data):
        # input_data: [B, N, C]
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data

        return hidden


class LinearizedSematicGraphConvolutional(nn.Module):
    def __init__(self, c_in, c_out, order=1):
        super(LinearizedSematicGraphConvolutional, self).__init__()
        gcn_in = c_out * 2
        self.fc_in = nn.Sequential(
            nn.Linear(c_in, gcn_in),
            nn.GELU()
        )
        c_hidden = (order + 1) * gcn_in
        self.fc_out = nn.Linear(c_hidden, c_out, bias=False)
        self.dropout = nn.Dropout(p=0.4)
        self.order = order
        self.act = nn.GELU()

    def linearized_gconv(self, x, nodevec):
        # AH
        hidden = torch.einsum("nc,bnh->bhc", nodevec, x)
        hidden = torch.einsum("nc,bhc->bnh", nodevec, hidden)

        # D
        degree = torch.ones([nodevec.shape[0]]).to(nodevec.device)
        degree = torch.einsum("nc,n->c", nodevec, degree)
        degree = torch.einsum("nc,c->n", nodevec, degree)
        degree = degree.unsqueeze(-1)

        # D^(-1)(A-I)H = AH/D - H/D
        hidden = hidden / degree - x / degree

        return hidden.contiguous()

    def forward(self, x, nodevec):
        # x: [B, N, C]

        x = self.fc_in(x)
        out = [x]

        # sematic graph convolutional
        x1 = self.linearized_gconv(x, nodevec)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.linearized_gconv(x1, nodevec)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.fc_out(h)
        h = self.act(h)
        h = self.dropout(h)

        return h


class MUGST(nn.Module):
    def __init__(self, args, num_nodes, transfer_matrix):
        super().__init__()

        # attributes
        self.num_coarse = args.num_coarse
        self.num_nodes = num_nodes
        self.spatial_dim = args.spatial_dim
        self.input_len = args.seq_len
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.embed_dim = args.embed_dim
        self.pred_len = args.pred_len
        self.num_layer = args.num_layer
        self.temp_dim_tid = args.temp_dim_tid
        self.temp_dim_diw = args.temp_dim_diw
        self.time_of_day_size = args.steps_per_day
        self.day_of_week_size = args.day_of_week_size
        self.gcn_dim = args.gcn_dim

        # ablation attributes
        self.spe_cluster = args.spe_cluster
        self.sem_gconv = args.sem_gconv
        self.fine = args.fine
        self.coarse = args.coarse

        self.transfer_matrix = transfer_matrix

        # input series embedding layer
        self.time_series_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len,
                                               out_features=self.embed_dim)

        # encoding
        ################################################################
        #                     fine-grained                              #
        ################################################################
        if self.fine:
            # spatial embeddings
            self.fine_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_dim))
            nn.init.xavier_uniform_(self.fine_emb)

            # temporal embeddings
            self.time_in_day_emb_fine = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb_fine)
            self.day_in_week_emb_fine = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb_fine)

            self.hidden_dim_fine = self.spatial_dim + \
                                   self.temp_dim_tid + self.temp_dim_diw + self.embed_dim
            self.node_mlp_layers = nn.ModuleList()
            hidden_dim_cur = self.hidden_dim_fine
            for _ in range(self.num_layer):
                self.node_mlp_layers.append(MultiLayerPerceptron(hidden_dim_cur))

            self.regression_layer_fine = nn.Linear(in_features=hidden_dim_cur,
                                                   out_features=self.pred_len)

        ################################################################
        #                     coarse-grained                          #
        ################################################################

        if self.coarse:
            # spatial embeddings
            self.coarse_emb = nn.Parameter(
                torch.empty(self.num_coarse, self.spatial_dim))
            nn.init.xavier_uniform_(self.coarse_emb)

            # temporal embeddings
            self.time_in_day_emb_coarse = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb_coarse)
            self.day_in_week_emb_coarse = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb_coarse)

            # nodevec transform
            if self.sem_gconv:
                self.emb_w1 = nn.Parameter(torch.randn(self.spatial_dim, self.spatial_dim))
                self.emb_w2 = nn.Parameter(torch.randn(self.spatial_dim, self.spatial_dim))

            self.hidden_dim_coarse = self.spatial_dim + \
                                     self.temp_dim_tid + self.temp_dim_diw + self.embed_dim
            self.coarse_mlp_layers = nn.ModuleList()
            self.sem_gconvs = nn.ModuleList()
            hidden_dim_cur = self.hidden_dim_coarse

            for _ in range(self.num_layer):
                self.coarse_mlp_layers.append(MultiLayerPerceptron(hidden_dim_cur))
                if self.sem_gconv:
                    self.sem_gconvs.append(LinearizedSematicGraphConvolutional(
                        c_in=hidden_dim_cur,
                        c_out=self.gcn_dim
                    ))
                    hidden_dim_cur = hidden_dim_cur + self.gcn_dim

            self.regression_layer_coarse = nn.Linear(in_features=hidden_dim_cur,
                                                     out_features=self.pred_len)

    def compute_adj(self, nodevec):
        adj = torch.mm(nodevec, nodevec.T)
        # no self-loop
        identity = torch.eye(nodevec.shape[0]).to(nodevec.device)
        return adj - identity

    def get_transfer_matrix(self, fine_emb, coarse_emb):
        return F.relu(torch.tanh(torch.mm(fine_emb, coarse_emb.T)))

    def forward(self, history_data):
        # history_data: [B, T, N, C]

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape
        # time series embedding
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1)
        # time in day
        t_i_d_data = history_data[..., -2]
        # day of week
        d_i_w_data = history_data[..., -1]

        time_series_emb = self.time_series_emb_layer(input_data)

        adj = None

        # fine-grained
        if self.fine:
            fine_emb = []
            fine_vec = F.dropout(self.fine_emb.unsqueeze(0), p=0.2, training=self.training)
            fine_emb.append(fine_vec.expand(batch_size, -1, -1))

            tem_emb_node = []
            time_in_day_emb_fine = self.time_in_day_emb_fine[
                (t_i_d_data[:, -1, 0:1] * self.time_of_day_size).type(torch.LongTensor)]
            tem_emb_node.append(time_in_day_emb_fine.expand(batch_size, self.num_nodes, -1))
            day_in_week_emb_fine = self.day_in_week_emb_fine[
                (d_i_w_data[:, -1, 0:1] * self.day_of_week_size).type(torch.LongTensor)]
            tem_emb_node.append(day_in_week_emb_fine.expand(batch_size, self.num_nodes, -1))

            # concate all embeddings
            hidden = torch.cat([time_series_emb] + fine_emb + tem_emb_node, dim=2)
            # fine-grained encoding
            for i in range(self.num_layer):
                hidden = self.node_mlp_layers[i](hidden)

            fine_regression = self.regression_layer_fine(hidden)
        else:
            fine_regression = 0.

        # coarse level
        if self.coarse:
            coarse_emb = []
            coarse_vec = F.dropout(self.coarse_emb.unsqueeze(0), p=0.2, training=self.training)
            coarse_emb.append(coarse_vec.expand(batch_size, -1, -1))

            # coarse temporal embeddings
            tem_emb_coarse = []
            time_in_day_emb_coarse = self.time_in_day_emb_coarse[
                (t_i_d_data[:, -1, 0:1] * self.time_of_day_size).type(torch.LongTensor)]
            tem_emb_coarse.append(time_in_day_emb_coarse.expand(batch_size, self.num_coarse, -1))
            day_in_week_emb_coarse = self.day_in_week_emb_coarse[
                (d_i_w_data[:, -1, 0:1] * self.day_of_week_size).type(torch.LongTensor)]
            tem_emb_coarse.append(day_in_week_emb_coarse.expand(batch_size, self.num_coarse, -1))

            # fine level -> coarse level
            if self.spe_cluster:
                transfer_matrix = self.transfer_matrix
            else:
                # get adaptive transaction matrix
                transfer_matrix = self.get_transfer_matrix(fine_vec.squeeze(), coarse_vec.squeeze())
                # visualize
                self.transfer_matrix = transfer_matrix
            time_series_emb_coarse = torch.matmul(time_series_emb.transpose(1, 2), transfer_matrix).transpose(1, 2)

            hidden_coarse = torch.cat([time_series_emb_coarse] + coarse_emb + tem_emb_coarse, dim=2)

            if self.sem_gconv:
                gate = F.softmax(torch.matmul(coarse_vec.squeeze(), self.emb_w1), dim=1)
                filter = F.relu(torch.matmul(coarse_vec.squeeze(), self.emb_w2))
                nodevec = F.normalize(gate * filter, p=2, dim=1)
                adj = self.compute_adj(nodevec)

            # coarse encoding
            for i in range(self.num_layer):
                hidden_coarse = self.coarse_mlp_layers[i](hidden_coarse)
                if self.sem_gconv:
                    hidden_gconv = self.sem_gconvs[i](hidden_coarse, nodevec)
                    hidden_coarse = torch.cat([hidden_coarse] + [hidden_gconv], dim=2)

            coarse_regression = self.regression_layer_coarse(hidden_coarse)
            # coarse level -> fine level
            coarse_regression = torch.matmul(coarse_regression.transpose(1, 2), transfer_matrix.T).transpose(1, 2)
        else:
            coarse_regression = 0.

        # aggregate all predictions
        prediction = fine_regression + coarse_regression
        prediction = prediction.transpose(1, 2).contiguous().unsqueeze(-1)  # [B, T, N, C]

        if self.training:
            return prediction, adj
        else:
            return prediction
