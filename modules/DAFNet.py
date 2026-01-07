import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel
from .ViT import ViT
from .DTEncoder import DisasterTextEncoderV2
from .EIEncoder import EditedImageEncoder
from .DCRFusion import DisasterConsensusReliabilityFusion
from .Classifier import DamageLevelClassifier
from .ViTrans import PretrainedViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.dim_model, config.num_classes)
        self.vit = ViT(config)
        self.dropout = nn.Dropout(config.dropout)
        self.dtencoder = DisasterTextEncoderV2(config.hidden_size)
        self.eiencoder = EditedImageEncoder(config.hidden_size, config.subspace_size, config.attn_hidden_size)
        self.dcrfusion = DisasterConsensusReliabilityFusion()
        self.classifier = DamageLevelClassifier()

        self.textd = nn.Linear(config.dim_model, 256)

        self.vitrans = PretrainedViT(model_path=config.vit_path)

    def vit_encoder(self, x):
        features = self.vitrans(x)
        cls = features['cls_token']
        img = features['img_token']
        return img, cls.squeeze(1)

    def text_en(self, x):
        out = self.textd(x)
        out = out.sum(dim=1)
        return out, 0

    def meanPool(self, dir, dir_mask):
        out, _ = self.bert(dir, attention_mask=dir_mask, output_all_encoded_layers=False)
        out = torch.mean(out, dim=1, keepdim=True).squeeze(1)
        return out


    def forward(self, x):
        img = x[0]
        context = x[1]  # 输入的句子
        dir = x[2]
        dir_mask = x[4]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[5]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        text, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        img, img_cls = self.vit_encoder(img)
        dir = self.meanPool(dir, dir_mask)
        text_, l_t_orth = self.dtencoder(text, mask, dir)
        # text_, l_t_orth = self.text_en(text)
        img_, l_i_orth = self.eiencoder(img)
        out= self.dcrfusion(text_, img_)
        out = self.dropout(out)
        out = self.classifier(out)
        return out, l_t_orth, l_i_orth