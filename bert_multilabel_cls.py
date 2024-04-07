# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        self.bert = BertModel.from_pretrained("bert-base-chinese")

    def forward(self, input_ids, attention_mask, token_type_ids,output_attentions=True):
        # 正确的调用方式
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=output_attentions)
        #outputs = self.bert(input_ids, attention_mask, token_type_ids,output_attentions)
        cls = self.drop(outputs[1])
        out = torch.sigmoid(self.fc(cls))
        if output_attentions:
            attentions = outputs[2]  # 注意力矩阵位于输出的第三个位置
            return out, attentions
        return out








