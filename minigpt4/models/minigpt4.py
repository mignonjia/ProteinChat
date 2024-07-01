import logging
import esm

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from transformers import AutoTokenizer, EsmModel
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
        freeze_protein_encoder=True,
        freeze_lp=False,
        freeze_llama=True,
        llama_model="",
        embedding_agg=1, 
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

        # esm_pt_path = "/nfs_baoding_ai/shuxian_2022/contact_prediction/esmfold_ckpt/esm2_t33_650M_UR50D.pt"
        # esm_pt_path = "/nfs_baoding_ai/shuxian_2022/contact_prediction/esmfold_ckpt/esm2_t36_3B_UR50D.pt"
        self.protein_encoder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # esm.pretrained.load_model_and_alphabet_local(esm_pt_path)
        self.protein_tokenizer = alphabet.get_batch_converter()
        # self.protein_encoder = self.protein_encoder.to(torch.cuda.current_device())

        # self.protein_tokenizer = AutoTokenizer.from_pretrained("/nfs_baoding_ai/huggingface/facebook/esm2_t33_650M_UR50D")
        # self.protein_encoder = EsmModel.from_pretrained("/nfs_baoding_ai/huggingface/facebook/esm2_t33_650M_UR50D")

        # if freeze_protein_encoder:
        #     self.args_.fp16 = True

        if freeze_protein_encoder:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            self.protein_encoder = self.protein_encoder.eval()
            self.protein_encoder.train = disabled_train
            logging.info("freeze protein encoder")
        else:
            self.protein_encoder = self.protein_encoder.train()
        
        print('Loading LLAMA')
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
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        self.glm_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size
        )
        if freeze_lp:
            for name, param in self.glm_llama_proj.named_parameters():
                param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # if prompt_path:
        #     with open(prompt_path, 'r') as f:
        #         raw_prompts = f.read().splitlines()
        #     filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<proteinHere>" in raw_prompt]
        #     self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        #     print('Load {} training prompts'.format(len(self.prompt_list)))
        #     print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        # else:
        #     self.prompt_list = []

    def encode_protein(self, seqs):
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(('protein', seq))
        batch_labels, batch_strs, batch_tokens = self.protein_tokenizer(batch_seqs)
        batch_tokens = batch_tokens.to(torch.cuda.current_device())
        # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations
        protein_embeds = self.protein_encoder(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33].to(batch_tokens.device)

        # inputs = self.protein_tokenizer(seqs, return_tensors="pt").to(torch.cuda.current_device())
        # outputs = self.protein_encoder(**inputs)

        # protein_embeds = outputs.last_hidden_state
        # print(f'Size of protein embedding: {protein_embeds.size()}')
        # print(protein_embeds)

        # input llama is of shape [B, len, 5120]
        if protein_embeds.dtype != self.glm_llama_proj.weight.dtype:
            protein_embeds = protein_embeds.to(self.glm_llama_proj.weight.dtype)

        inputs_llama = self.glm_llama_proj(protein_embeds.squeeze(dim=2)).to(protein_embeds.device)
        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(protein_embeds.device)
        #print(f'Size of inputs_llama: {inputs_llama.size()}')
        #print(f'Size of atts_llama: {atts_llama.size()}')
        return inputs_llama, atts_llama

    def prompt_list_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            p_before_lst = []
            p_after_lst = []
            for p in prompt:
                p_before, p_after = p.split('<proteinHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)
            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(img_embeds.device)
            
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens_lst.input_ids)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        seqs = samples["seq"] # list of seq
        # print(samples)
        protein_embeds, atts = self.encode_protein(seqs)

        img_embeds, atts_img = self.prompt_list_wrap(protein_embeds, atts, samples["prompt"])

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

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
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
        #print("===========")
        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):

        llama_model = cfg.get("llama_model")

        freeze_protein_encoder = cfg.get("freeze_protein_encoder", False)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        model = cls(
            freeze_protein_encoder=freeze_protein_encoder,
            freeze_lp=freeze_lp,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            embedding_agg = embedding_agg, 
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
