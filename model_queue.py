
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils_HSI import get_device
import argparse
# device_num=get_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device")
args = parser.parse_args()
device_num=get_device(args.cuda)
# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True) 
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, CLASS_NUM, patch_size, n_bands, embed_dim):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2), padding=(0,1,1))
        self.conv1 = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=(1,3,3), bias=False)
        self.patch_size = patch_size
        # self.final_feat_dim = 128
        self.fc = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)
        num = CLASS_NUM.astype(int)
        self.classifier = nn.Linear(in_features=self._get_layer_size(), out_features=num, bias=False)

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, self.n_bands,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0],-1)
            s = x.size()[1]
        return s

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.block1(x) 
        x = self.maxpool1(x) 
        x = self.block2(x) 
        x = self.maxpool2(x) 
        x = self.conv1(x) 
        x = x.view(x.shape[0],-1)
        y = self.classifier(x)
        proj = self.fc(x)
        return y, proj


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

####################################################  VAE_img  #######################################################################
class VAE_img(nn.Module):
    def __init__(self):
        super(VAE_img, self).__init__()
        self.fc1 = nn.Linear(33462, 4096)  # SH
        self.fc12 = nn.Linear(4096, 1024)  # SH
        # self.fc1 = nn.Linear(17238, 1024)  # UP
        self.fc2 = nn.Linear(1024,256)    # UP SH
        self.fc21 = nn.Linear(256, 64)
        self.fc22 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc32 = nn.Linear(256,1024)    # UP  SH
        self.fc33 = nn.Linear(1024,4096)   # SH
        # self.fc4 = nn.Linear(1024, 17238)  # UP
        self.fc4 = nn.Linear(4096, 33462)  # SH

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc12(h1)) 
        h1 = F.relu(self.fc2(h1))  # UP
        mu = F.relu(self.fc21(h1))
        logvar = F.relu(self.fc22(h1))       
        return mu , logvar 

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar)
        #eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).to(device_num)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc32(h3))   # UP
        h3 = F.relu(self.fc33(h3))   # UP
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):    # x:(256,48,13,13)  up(256,102,13,13)
        x = x.view(x.shape[0], -1)  # (256,8112)Houston   UP(256,17238)  SH(256,33462)
        mu, logvar = self.encode(x)  #(256,64)
        z = self.reparametrize(mu, logvar) #(256,64)
        rec = self.decode(z)  #(256,8112)Houston   UP(256,17238)  SH(256,33462)
        rec = rec.reshape((256,198,13,13))  # SH
        # rec = rec.reshape((256,102,13,13))  # UP
        return rec, z, mu, logvar

####################################################  VAE_tex  #######################################################################
class VAE_tex(nn.Module):
    def __init__(self):
        super(VAE_tex, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc21 = nn.Linear(256, 64)
        self.fc22 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc33 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc11(h1))
        mu = F.relu(self.fc21(h1))
        logvar = F.relu(self.fc22(h1))
        
        return mu , logvar  #

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar)
        #eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).to(device_num)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def decode(self, z):
        # 原始解码器
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
    
        # 原始解码器
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc33(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # cls = self.classifier(z)
        return self.decode(z), z, mu, logvar


###########################################################################################################################################


class LDGnet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 inchannel,
                 vision_patch_size: int,
                 num_classes,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length


        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.visual = D_Res_3d_CNN(1,8,16,num_classes, vision_patch_size, inchannel, embed_dim)
        self.initialize_parameters()

        # VAE_tex
        self.vae_tex = VAE_tex()
        self.vae_img = VAE_img()
        self.fusion = nn.Linear(1024,512)
        self.reconstruction_criterion = nn.MSELoss(size_average=False)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((13, 13))
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mode):
        # return self.visual(image.type(self.dtype), mode)
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, current_epoch, image, text, label, text_queue_1=None, text_queue_2=None):
        image = self.adaptive_pool(image)
        imgage_prob, image_features = self.encode_image(image, mode='train')

        if self.training:  # image_prob: (256,7)   image_features:(256,512)

            img_rec, z_img, mu_img, logvar_img = self.vae_img(image)
            imgage_prob_rec, image_features_rec = self.encode_image(img_rec, mode='train')

            text_features = self.encode_text(text)  # 粗粒度文本  (256,512)
            text_features_q1 = self.encode_text(text_queue_1)  # 细粒度文本1
            text_features_q2 = self.encode_text(text_queue_2)  # 细粒度文本2

            reconstruction_loss = self.reconstruction_criterion(img_rec, image)
            KLD = (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

            # 自适应调节参数
            self.current_epoch = current_epoch
            self.warmup = {'beta': {'factor': 0.25,
                                    'end_epoch': 93,
                                    'start_epoch': 0},
                           'distance': {'factor': 8.13,
                                        'end_epoch': 22,
                                        'start_epoch': 0}}
            f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (
                    1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
            f2 = f2 * (1.0 * self.warmup['beta']['factor'])
            beta = torch.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])]).to(device_num)
              # # VAE损失
            fac = 5e-4
            loss_VAE = (reconstruction_loss - beta * KLD) * fac
            image_features = image_features / image_features.norm(dim=1, keepdim=True)  # 对张量中的元素进行归一化处理
            text_features = text_features / text_features.norm(dim=1, keepdim=True)  # (256,512)

            # cosine similarity as logits
            ####################################################注释1#############################################################
            logit_scale = self.logit_scale.exp()  # logit_scale是一个实数值，用于调整图像特征和文本特征之间点积（即相似度矩阵）的范围
            # 特征提取之后，由于做了normalize，直接相乘来计算余弦距离，同一pair对的结果趋近于1，不同pair对的结果趋近于0
            logits_per_image = logit_scale * image_features @ text_features.t()  # (256,256)
            logits_per_text = logit_scale * text_features @ image_features.t()

            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_clip = (loss_img + loss_text) / 2
            ######################################################################################################################

            text_features_q1 = text_features_q1 / text_features_q1.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            ####################################################注释2#############################################################
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features_q1.t()
            logits_per_text = logit_scale * text_features_q1 @ image_features.t()

            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_q1 = (loss_img + loss_text) / 2

            text_features_q2 = text_features_q2 / text_features_q2.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            ####################################################注释3#############################################################
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features_q2.t()
            logits_per_text = logit_scale * text_features_q2 @ image_features.t()

            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            loss_q2 = (loss_img + loss_text) / 2
            return loss_clip, (loss_q1 + loss_q2) / 2, imgage_prob, loss_VAE, imgage_prob_rec
        else:
            return torch.tensor(0).long(), imgage_prob

