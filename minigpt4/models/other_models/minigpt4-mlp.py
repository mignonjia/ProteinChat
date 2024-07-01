# run baseline for classification
import logging

#import sys
#sys.path.append('/nfs/mingjia/proteinchat_seq/anti_1b_code')

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train

from anti_1b_code.estimated_ppl import get_embedding, initialize_model_and_tokenizer
# from anti_1b_code.initialize import 
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import time

q_id = 5
answer_list_all = [
    ["Molecule Transport", "Transcription from DNA to mRNA","Amino-acid biosynthesis","Protein biosynthesis from mRNA molecules","Lipid metabolism","tRNA processing","DNA damage","Cell cycle"],
    ["NAD", "NADP"],
    ["Nucleotide", "Magnesium", "Zinc", "Iron", "S-adenosyl-L-methionine", "Manganese"],
    ["Transferase", "Hydrolase", "Oxidoreductase", "Ligase", "Lyase", "Isomerase", "Translocase"],
    ["Ribonucleoprotein", "Chaperone protein"],
    ["Membrane", "Secreted", "Plastid", "Cytoplasm", "Nucleus", "Mitochondrion"],
    ["NO", "YES"],
    ["NO", "YES"]
]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

        torch.nn.init.kaiming_normal_(self.layer1.weight)
        torch.nn.init.zeros_(self.layer1.bias)
        torch.nn.init.kaiming_normal_(self.layer2.weight)
        torch.nn.init.zeros_(self.layer2.bias)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        # vit_model="eva_clip_g",
        # q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        freeze_protein_encoder=True,
        freeze_lp = False,
        freeze_llama = True,
        # freeze_qformer=True,
        # num_query_token=32,
        llama_model="",
        embedding_agg=1, 
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        print("Q_id minigpt4:", q_id)

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        
        print('Loading protein encoder')
        parser = ArgumentParser()
        self.args_ = parser.parse_args()
        with open('train_configs/glm_config.txt', 'r') as f:    
            self.args_.__dict__ = json.load(f)
        
        self.args_.device = torch.cuda.current_device()

        if freeze_protein_encoder:
            self.args_.fp16 = True

        self.protein_encoder, self.protein_tokenizer = initialize_model_and_tokenizer(self.args_, freeze_protein_encoder)
   
        if freeze_protein_encoder:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            self.protein_encoder = self.protein_encoder.eval()
            self.protein_encoder.train = disabled_train
            logging.info("freeze protein encoder")
        # else:
            # Debug, check model precision
            # print("GLM 130 B parameters")
            # for param in self.protein_encoder.parameters():
            #     print(param.dtype)
        #
        # print('Loading Q-Former')
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)
        #
        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.query_tokens.requires_grad = False
        #     logging.info("freeze Qformer")
        # print('Loading Q-Former Done')
        
        self.num_c = len(answer_list_all[q_id])
        self.mlp = MLP(2048, 128, self.num_c)
        print("Number of choices", self.num_c)

        if freeze_lp:
            for name, param in self.mlp.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_protein(self, seqs):
        protein_embeds = get_embedding(seqs, self.args_, self.embedding_agg, self.protein_encoder, self.protein_tokenizer)
        protein_embeds = torch.mean(protein_embeds, 1)
        if protein_embeds.dtype != self.mlp.layer1.weight.dtype:
            protein_embeds = protein_embeds.to(self.mlp.layer1.weight.dtype)
        # input llama is of shape [B, len, 5120]
        # if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
        #     protein_embeds = protein_embeds.to(self.glm_llama_proj.weight.dtype)

        return protein_embeds


    def forward(self, samples):
        criterion = nn.CrossEntropyLoss()
        seqs = samples["seq"] # list of seq
        y = samples["answer_id"]
        protein_embeds = self.encode_protein(seqs)
        logits = self.mlp(protein_embeds)
        loss = criterion(logits, y)

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):

        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        # img_size = cfg.get("image_size")
        # num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")
        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        freeze_protein_encoder = cfg.get("freeze_protein_encoder", True)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        #print(embedding_agg)

        model = cls(
            # vit_model=vit_model,
            # q_former_model=q_former_model,
            # img_size=img_size,
            # drop_path_rate=drop_path_rate,
            # use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            freeze_protein_encoder=freeze_protein_encoder,
            freeze_llama=freeze_llama,
            freeze_lp=freeze_lp,
            # freeze_qformer=freeze_qformer,
            # num_query_token=num_query_token,
            llama_model=llama_model,
            embedding_agg = embedding_agg, 
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )


        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and LP
        if stage1_ckpt:
            print("Load GLM and LP Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
