import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_layers import GraphResBlock, GraphLinear, spmm
from metro_layers.modeling_metro import *
from metro_layers.modeling_bert import BertConfig
from pointnet2 import *
from chamfer_distance import chamfer_dists
from configuration.config import *
import transformer

# https://github.com/jiashunwang/Long-term-Motion-in-3D-Scenes/blob/f52948ff9ba30c7d938b5cf4c2cc3e254b9cbb49/sub_goal.py#L19
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        #self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(256)
        #if self.feature_transform:
        #    self.fstn = STNkd(k=64)

    def forward(self, x):
        x = x[:, :3, :]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        return x

class PointNet2feat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, freeze=False):
        super(PointNet2feat, self).__init__()
        self.encoder = LocalPointEncoder(freeze=freeze)
        self.conv = nn.Conv1d(512, 16, 1)

    def forward(self, x):
        x = x[:, :9, :]
        xyz, points = self.encoder(x)
        points = self.conv(points)
        return torch.flatten(torch.cat([xyz, points], dim=1), start_dim=1)

class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()
        self.num_feat_in = num_feat_in
        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        Xout = Xout.reshape(-1, self.num_feat_in)
        # return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))
        return torch.distributions.normal.Normal(self.mu(Xout), torch.exp(0.5 * self.logvar(Xout)))

class ResBlock(nn.Module):
    def __init__(self, n_dim):
        super(ResBlock, self).__init__()
        self.n_dim = n_dim

        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.acfun = nn.LeakyReLU()

    def forward(self, x0):
        x = self.acfun(self.fc1(x0))
        x = self.acfun(self.fc2(x))
        x = x + x0
        return x

class TransformVAE(nn.Module):
    def __init__(self, args):
        super(TransformVAE, self).__init__()
        self.args = args
        num_channels = args.num_channels
        self.obj_encoder = nn.Sequential(
            PointNetfeat(),
            nn.Linear(256, num_channels),
            # PointNet2feat(freeze=args.freeze),
            # nn.Linear(304, num_channels),
        )
        self.body_linear = nn.Linear(9 + args.verb_dim, num_channels)
        self.encoder = nn.Sequential(
            ResBlock(n_dim=2 * num_channels),
            ResBlock(n_dim=2 * num_channels)
        )
        self.latent_normal = NormalDistDecoder(2 * num_channels, args.latent_dimension)
        self.latent_linear = nn.Linear(args.latent_dimension + args.verb_dim, num_channels)
        self.decoder = nn.Sequential(
            ResBlock(n_dim=2 * num_channels),
            ResBlock(n_dim=2 * num_channels),
            nn.Linear(2 * num_channels, 9),
        )

    def forward(self, frame, obj_points, verb_code):
        batch_size = frame.shape[0]

        body_code = self.body_linear(torch.cat([frame.reshape(batch_size, -1), verb_code], dim=1))
        obj_code = self.obj_encoder(obj_points.transpose(1, 2))
        if self.args.dummy_obj:
            obj_code = obj_code * 0
        encoder_input = torch.cat([body_code, obj_code], dim=1)
        feature = self.encoder(encoder_input)
        z_dist = self.latent_normal(feature)

        z_sample = z_dist.rsample()
        # print(z_sample.shape, verb_code.shape, obj_code.shape)
        decoder_input = torch.cat([self.latent_linear(torch.cat([z_sample, verb_code], dim=1)),
                                   obj_code], dim=1)
        rec = self.decoder(decoder_input)
        return rec, z_dist

    def sample(self, obj_points, verb_code):
        set_eval = self.training
        if set_eval:
            self.eval()
        with torch.no_grad():
            batch_size = obj_points.shape[0]
            obj_code = self.obj_encoder(obj_points.transpose(1, 2))
            if self.args.dummy_obj:
                obj_code = obj_code * 0
            z_sample = torch.distributions.normal.Normal(
                loc=torch.zeros((batch_size, self.args.latent_dimension), requires_grad=False, device=self.args.device),
                scale=torch.ones((batch_size, self.args.latent_dimension), requires_grad=False,
                                 device=obj_points.device)).rsample()
            decoder_input = torch.cat([self.latent_linear(torch.cat([z_sample, verb_code], dim=1)),
                                       obj_code], dim=1)
            rec = self.decoder(decoder_input)
        if set_eval:
            self.train()
        return rec

def padded_idx_to_code(verb_ids):
    codebook = torch.cat([torch.eye(4, dtype=torch.float32, device=verb_ids.device),
               torch.zeros((1, 4), dtype=torch.float32, device=verb_ids.device)], dim=0)
    return codebook[verb_ids.long()]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    # input: BxSxD
    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class ObjEncoder(nn.Module):
    def __init__(self, num_channels, ):
        super(ObjEncoder, self).__init__()
        self.num_channels = num_channels
        self.pointnet = nn.Sequential(
            PointNetfeat(),
            nn.Linear(256, num_channels),
            # PointNet2feat(freeze=args.freeze),
            # nn.Linear(304, num_channels),
        )

    def forward(self, input):
        B, I, P, C = input.shape
        output = self.pointnet(input.reshape(B*I, P, C).transpose(1, 2)) # B*IxD
        return output.reshape(B, I, 1, self.num_channels)

class Pointnet2Encoder(nn.Module):
    def __init__(self, out_dimension, freeze=False):
        super(Pointnet2Encoder, self).__init__()
        self.out_dimension = out_dimension
        self.out_points = 16
        self.pointnet2 = nn.Sequential(
            LocalPointEncoder(freeze=freeze),
        )
        self.linear = nn.Linear(512 + 3, out_dimension)

    def forward(self, obj_points):
        B, I, P, C = obj_points.shape
        xyz, points = self.pointnet2(
            obj_points.permute((0, 1, 3, 2)).reshape(B * I, C, P),
        )
        obj_embedding = self.linear(torch.cat([xyz, points], dim=1).transpose(1, 2)).reshape(B, I, self.out_points,
                                                                                             self.out_dimension)
        return obj_embedding

class CompositeTransformVAE(nn.Module):
    def __init__(self, args):
        super(CompositeTransformVAE, self).__init__()
        self.args = args
        num_channels = args.num_channels
        self.bodyEmbedding = nn.Linear(9, num_channels)
        # self.objEmbedding = ObjEncoder(num_channels=num_channels)
        # self.objEmbedding = Pointnet2Encoder(freeze=args.freeze, out_dimension=num_channels)
        self.objEmbedding = nn.Linear(9, num_channels)
        self.interactionEmbedding = nn.Parameter(torch.randn(4, num_channels))
        self.positionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)

        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        if args.return_attention:
            from transformer import TransformerDecoderLayer, TransformerDecoder
        else:
            from torch.nn import TransformerDecoderLayer, TransformerDecoder
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=num_channels,
                                                          nhead=self.args.num_heads,
                                                          dim_feedforward=self.args.ff_size,
                                                          dropout=self.args.dropout,
                                                          activation=self.args.activation,
                                                          batch_first=True)
        self.encoder = TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.args.num_layers)

        self.latentNormal = NormalDistDecoder(num_channels, args.latent_dimension)
        self.interactionBias = nn.Parameter(torch.randn(4, args.latent_dimension))
        self.latentLinear = nn.Linear(args.latent_dimension, num_channels)

        seqTransDecoderLayer = TransformerDecoderLayer(d_model=num_channels,
                                                          nhead=self.args.num_heads,
                                                          dim_feedforward=self.args.ff_size,
                                                          dropout=self.args.dropout,
                                                          activation=self.args.activation,
                                                          batch_first=True)
        self.decoder = TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.args.num_layers)
        self.finalLinear = nn.Linear(num_channels, 9)

    def get_embeddings(self, x=None, batch=None):
        num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
        B, I, _, _ = obj_points.shape
        Po, Pb, D = self.args.num_obj_keypoints, 1, self.args.num_channels
        padding_mask = (verb_ids == -1).repeat_interleave(Po).reshape((B, Po * 2))  # BxPo*2
        padding_mask = torch.cat([torch.zeros((B, Pb), dtype=torch.bool, device=obj_points.device),
                                  padding_mask], dim=1)
        verb_codes = padded_idx_to_code(verb_ids)  # Bx2x4
        # print(padding_mask)
        # print(verb_codes)
        verb_embedding = torch.matmul(verb_codes, self.interactionEmbedding)  # Bx2xD
        interaction_embedding = (verb_embedding.sum(dim=1) / num_atomics.unsqueeze(1)).unsqueeze(1)

        body_embedding = None
        if x is not None:
            x = x.reshape(B, Pb, -1)
            body_embedding = self.bodyEmbedding(x)  # BxPbx9 -> BxPbxD
            # body_embedding = self.positionalEmbedding(body_embedding)
            # body_embedding = body_embedding + interaction_embedding
        obj_embedding = self.objEmbedding(obj_points)  # Bx2xPoxD
        obj_embedding = obj_embedding + verb_embedding.unsqueeze(2)
        # print(obj_embedding.shape)
        obj_embedding = obj_embedding.reshape(B, I * Po, D)
        template_embedding = torch.zeros((B, Pb, D), device=obj_points.device)
        # template_embedding = self.positionalEmbedding(template_embedding)
        # template_embedding = template_embedding + interaction_embedding

        return verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding

    def forward(self, x, batch):
        num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
        Pb = 1
        verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self.get_embeddings(x=x, batch=batch)

        encoder_input = torch.cat([body_embedding, obj_embedding], dim=1)  # BxPb+2PoxD
        feature = self.encoder(encoder_input, src_key_padding_mask=padding_mask)[:, :Pb, :].mean(dim=1)
        z_dist = self.latentNormal(feature)  # BxLatent

        z_sample = z_dist.rsample()
        interaction_bias = torch.matmul(verb_codes, self.interactionBias).sum(dim=1) / num_atomics.unsqueeze(1)
        # z_sample = self.latentLinear(z_sample + interaction_bias).unsqueeze(1)  #Bx1xD
        z_sample = self.latentLinear(z_sample).unsqueeze(1)  # Bx1xD
        decoder_input = torch.cat([template_embedding, obj_embedding], dim=1)  # BxPb+2*PoxD
        decoder_output = self.decoder(tgt=decoder_input, memory=z_sample, tgt_key_padding_mask=padding_mask)
        pred = decoder_output[0] if self.args.return_attention else decoder_output
        pred = self.finalLinear(pred[:, 0, :])  # Bx9

        return pred, z_dist

    def sample(self, batch):
        set_eval = self.training
        if set_eval:
            self.eval()
        with torch.no_grad():
            num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
            B, I, _, _ = obj_points.shape
            Po, Pb = self.args.num_obj_keypoints, 1
            verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self.get_embeddings(x=None, batch=batch)

            z_sample = torch.distributions.normal.Normal(
                loc=torch.zeros((B, self.args.latent_dimension), requires_grad=False, device=obj_points.device),
                scale=torch.ones((B, self.args.latent_dimension), requires_grad=False,
                                 device=obj_points.device)).rsample()
            interaction_bias = torch.matmul(verb_codes, self.interactionBias).sum(dim=1) / num_atomics.unsqueeze(1)
            # z_sample = self.latentLinear(z_sample + interaction_bias).unsqueeze(1)  #Bx1xD
            z_sample = self.latentLinear(z_sample).unsqueeze(1)  # Bx1xD
            decoder_input = torch.cat([template_embedding, obj_embedding], dim=1)  # BxPb+2*PoxD
            decoder_output = self.decoder(tgt=decoder_input, memory=z_sample, tgt_key_padding_mask=padding_mask)
            pred = decoder_output[0] if self.args.return_attention else decoder_output
            pred = self.finalLinear(pred[:, 0, :])  # Bx9
        if set_eval:
            self.train()
        if self.args.return_attention:
            self_attention = decoder_output[1]
            return pred, self_attention
        else:
            return pred
