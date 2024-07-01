import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from einops import rearrange, reduce, repeat
from minigpt4.common.registry import registry
import numpy as np


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following protein: <protein>proteinContent</protein>. "
           "Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)



class Chat:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        # self.model.llama_model = self.model.llama_model.bfloat16()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</protein>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)
        
        # print("===after ask:", conv)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, save_embeds=False):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        if save_embeds:
            # print(embs.squeeze().detach().cpu().numpy().shape)
            np.save('/nfs_baoding_ai/mingjia_2023/proteinchat_glm/tsne/prompt.npy', embs.squeeze().detach().cpu().numpy())

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        # print("num_beams, temperature", num_beams, temperature)
        with self.model.maybe_autocast():   
            outputs = self.model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=False,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=float(temperature),
                output_hidden_states=False
            )
        with torch.no_grad():
            results = self.model.llama_model(input_ids=outputs, labels=outputs)
            neg_log_likelihood = results.loss.item()
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]

        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text

        # if save_embeds:
        #     print(outputs.hidden_states[-1].squeeze().detach().cpu().numpy())
        #     print(outputs.hidden_states[-1].detach().cpu().numpy().shape)
        return output_text, output_token.cpu().numpy(), neg_log_likelihood

    def get_ppl(self, conv, img_list, predict_list):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        
        predict_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=False).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(predict_list)
        ]
        predict_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in predict_tokens]

        conf_list = []
        with torch.no_grad():
            for id_pred in range(len(predict_embs)):
                # predict_emb = predict_embs[id_pred]
                # input_emb = torch.cat([embs, predict_emb], dim=1)
                # tokd_labels = predict_tokens[id_pred]
                # tokd_prefix = torch.full((tokd_labels.shape[0], input_emb.shape[1] - tokd_labels.shape[1] + 2), -100).cuda()
                # tokd_labels = torch.cat([tokd_prefix, tokd_labels[:,2:]], dim=1)

                if id_pred == 0:
                    predict_emb = predict_embs[0]
                else:
                    predict_emb = torch.cat(predict_embs[:(id_pred+1)], dim=1)
                # print("predict_emb.shape", predict_emb.shape)
                input_emb = torch.cat([embs, predict_emb], dim=1)

                tokd_labels = predict_tokens[id_pred]
                # print("tokd_labels.shape before concat", tokd_labels.shape)
                tokd_prefix = torch.full((tokd_labels.shape[0], input_emb.shape[1] - tokd_labels.shape[1] + 1), -100).cuda()
                tokd_labels = torch.cat([tokd_prefix, tokd_labels[:,1:]], dim=1)
                # print("tokd_labels.shape after concat", tokd_labels.shape)

                results = self.model.llama_model(inputs_embeds=input_emb, labels=tokd_labels)
                neg_log_likelihood = results.loss.item()
                # print(neg_log_likelihood)
                # print("=================")
                conf_list.append(neg_log_likelihood)

        # exit()

        return conf_list


    def upload_protein(self, seq, conv, protein_list):
        protein_emb, _ = self.model.encode_protein([seq])
        #protein_emb = rearrange(protein_emb, 't b c -> b t c')
        protein_list.append(protein_emb)
        conv.append_message(conv.roles[0], "<protein><proteinHere></protein>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return protein_emb, msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        # print("===prompt:", prompt)
        prompt_segs = prompt.split('<proteinHere>')
        # print("prompt_segs", prompt_segs)
        # print("=====")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of protein placeholders and proteins."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # for seg in seg_tokens:
        #     print("seg_token", seg)
        #     print("seg_token", seg.shape)
        # print("=====")
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # for seg in seg_embs:
        #     print("seg_emb", seg.shape)
        # print("=====")
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # for seg in mixed_embs:
        #     print("mixed_emb", seg.shape)
        # print("=====")
        mixed_embs = torch.cat(mixed_embs, dim=1)
        # print(mixed_embs.shape)
        return mixed_embs


