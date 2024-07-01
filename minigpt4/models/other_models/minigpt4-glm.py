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
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from anti_1b_code.estimated_ppl import get_embedding, initialize_model_and_tokenizer
# from anti_1b_code.initialize import 
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import time

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
        
        if "v1.5" in llama_model:
            print('Loading Vicuna based on LLaMA 2')
        else:
            print('Loading Vicuna based on LLaMA 1')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
                # device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # print(self.llama_model.model)
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        self.glm_llama_proj = nn.Linear(
            2048, self.llama_model.config.hidden_size
        )
        
        if freeze_lp:
            for name, param in self.glm_llama_proj.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_protein(self, seqs):
        protein_embeds = get_embedding(seqs, self.args_, self.embedding_agg, self.protein_encoder, self.protein_tokenizer)
        # input llama is of shape [B, len, 5120]
        if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
            protein_embeds = protein_embeds.to(self.glm_llama_proj.weight.dtype)

        inputs_llama = self.glm_llama_proj(protein_embeds.squeeze(dim=2)).to(protein_embeds.device)
        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(protein_embeds.device)
        #print(f'Size of inputs_llama: {inputs_llama.size()}')
        #print(f'Size of atts_llama: {atts_llama.size()}')
        return inputs_llama, atts_llama

    def prompt_list_wrap(self, protein_embeds, atts_img, prompt):
        if prompt:
            p_before_lst = []
            p_after_lst = []
            for p in prompt:
                p_before, p_after = p.split('<proteinHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)
            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(protein_embeds.device)
            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(protein_embeds.device)
            
            # lst = ['Describe this protein.', 'De', 'scribe', 'Des', 'cribe', 'Desc', 'ribe', 'Descr', 'ibe', 'Descri', 'be', 'Describ', 'e']
            # for i in range(len(lst)):
            #     ret = self.llama_tokenizer(
            #         lst[i], return_tensors="pt", add_special_tokens=True, padding=True).to(protein_embeds.device)
            #     print(lst[i])
            #     print(ret)
            # exit()
            # p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens_lst.input_ids)
            # p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens_lst.input_ids)
            
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens_lst.input_ids)

            wrapped_protein_embeds = torch.cat([p_before_embeds, protein_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_protein_embeds.shape[1])
            return wrapped_protein_embeds, wrapped_atts_img
        else:
            return protein_embeds, atts_img

    def forward(self, samples):
        seqs = samples["seq"] # list of seq
        # print(samples)
        protein_embeds, atts = self.encode_protein(seqs)

        prompt_embeds, atts_img = self.prompt_list_wrap(protein_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(protein_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(protein_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = prompt_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        
        # bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_embeds = self.llama_model.get_input_embeddings()(bos)

        atts_bos = atts_img[:, :1]

        # to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        logits = outputs.logits
        # print(torch.argmax(logits, dim=2).shape)
        logits = torch.argmax(logits, dim=2)
        #print(self.llama_tokenizer.batch_decode(logits, skip_special_tokens=True)[-400:])

        loss = outputs.loss
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
