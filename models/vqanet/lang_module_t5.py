import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from models.transformer.attention import MultiHeadAttention
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from .modified_t5 import T5Model

from lib.configs.config_vqa import CONF
#对CONF配置对象赋值方式：手动。配置文件加载。命令行参数传递argparse。代码中动态赋予（条件语句）

class LangModule(nn.Module):  # num_text_classes: question id types
    def __init__(self, num_text_classes=100, answer_class_number=2000, model_name='t5', pretrained_path='/mnt/lustre/zhaolichen/codes/3dvl/t5_base',
                 use_lang_classifier=True, hidden_size=768, obj_sem_cls_class_number = 18):
        super().__init__()

        self.model_name = model_name
        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier

        if model_name in ['t5']:
            self.model = T5Model()
        else:
            raise NotImplementedError(model_name)


        #nn.Linear线性层（全连接层）作用是特征映射，两个参数含义分别是输入特征和输出特征             
        # language classifier
        if use_lang_classifier:
            self.query_cls = nn.Linear(hidden_size, num_text_classes)
            self.answering_fc = nn.Linear(hidden_size, answer_class_number)
            self.lang_sem_cls_class = nn.Linear(hidden_size, obj_sem_cls_class_number * len(CONF.OBJ_TYPE))

    def encode(self, data_dict):
        """
        encode the input descriptions
        """
        #第二维：每个序列token数；第三维：描述或问题的最大长度（用于可变长度序列）
        word_embs = data_dict["vqa_question_embedding"]  # B * 32 * MAX_DES_LEN * LEN(300)
        lang_len = data_dict["vqa_question_embedding_length"]
        device = word_embs.device

        # todo: there are xx objects in the scene (more info)
        query, answer = data_dict['vqa_question'], data_dict['vqa_answer']
        batch_size, lang_num_max = len(query), len(query[0])
        #将嵌套列表展开成扁平化一维列表
        query = [x for y in query for x in y]
        answer = [x for y in answer for x in y]

        tokens = self.model.tokenize_forward(query, answer, device)
        encoder_outputs = self.model.encoder_forward(input_ids=tokens['input_ids'], return_dict=True)
        hidden_state = encoder_outputs.last_hidden_state  # same as decoder_outputs.encoder_last_hidden_state
        data_dict['tokens'] = tokens
        data_dict['t5_encoder_outputs'] = encoder_outputs
        data_dict['vqa_question_attention_mask'] = None
        #存储了 VQA 问题的语言特征，这是从编码器的最后一层隐藏状态中提取的
        data_dict['vqa_question_lang_fea'] = hidden_state
        return data_dict

    def decode(self, data_dict):
        query, answer = data_dict['vqa_question'], data_dict['vqa_answer']
        batch_size, lang_num_max = len(query), len(query[0])
        tokens = data_dict['tokens']
        encoder_outputs = data_dict['t5_encoder_outputs']
        updated_lang_fea = data_dict['updated_lang_fea']
        # use the updated language feature
        encoder_outputs.hidden_state = updated_lang_fea

        # Note: The input tokens are trained with an seq-to-seq model, so we could only use the hidden_state[:, 0, :]
        outputs = self.model.decoder_forward(labels=tokens['labels'], return_dict=True, encoder_outputs=encoder_outputs, output_hidden_states=True)

        # embs = self.model.generate(**tokens)
        # import ipdb; ipdb.set_trace();
        # lang_feat_mask = torch.zeros_like(query_tokens).bool()
        # lang_feat_mask[query_tokens == self.tokenzier.tokenizer.pad_token_id] = True

        loss, logits, hidden_state = outputs.loss, outputs.logits, outputs.decoder_hidden_states[-1]
        #选择每个位置上概率最大的类别，生成一个包含类别索引的列表的列表
        prediction_answer = [[x for x in y] for y in logits.argmax(-1)]
        #使用模型的 tokenizer 对每个类别索引进行解码，将其转换为对应的字符串表示。
        prediction_answer = [self.model.tokenizer.decode(x) for x in prediction_answer]  # string

        # import ipdb; ipdb.set_trace()
        # cap_emb = hidden_state[:, 0, :]
        # store the encoded language features
        #存储整个问题的编码语言特征，并提取每个样本中最后一个时间步的特征，并以三维张量的形式存储在 last_feat 中
        data_dict["vqa_question_lang_decoded_fea"] = hidden_state  # B, hidden_size
        last_feat = hidden_state[:, 0, :].reshape(batch_size, lang_num_max, -1)

        # classify
        if self.use_lang_classifier:
            # We Only Use Feature[0] for classification (seq2seq, cls token)
            data_dict["vqa_question_lang_scores"] = self.query_cls(last_feat)

            pred_answer = self.answering_fc(last_feat)
            pred_lang_sem_cls = self.lang_sem_cls_class(last_feat).reshape(batch_size, lang_num_max, -1, len(CONF.OBJ_TYPE))
            data_dict["vqa_pred_answer"] = pred_answer
            data_dict["vqa_pred_lang_sem_cls"] = pred_lang_sem_cls

        return data_dict

    def forward(self, data_dict):
        raise NotImplementedError('Cross-Modal-Attention is needed!')

