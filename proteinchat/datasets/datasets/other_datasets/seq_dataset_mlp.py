import os
import sys
from proteinchat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random

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

class SeqDataset(BaseDataset):
    def __init__(self, kw_path, text_rule_path, text_manual_path, seq_path):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        kw_entrys = json.load(open(kw_path, "r")) 
        self.sequence = json.load(open(seq_path, "r"))

        answer_list = answer_list_all[q_id]
        self.kw = [entry for entry in kw_entrys if entry['Q_id'] == q_id and entry['A'] in answer_list]

        self.answer_map = {}
        for i in range(len(answer_list)):
            self.answer_map[answer_list[i]] = i
        print("Q_id dataset", q_id)

        return

    def __len__(self):
        return len(self.kw)

    def __getitem__(self, index):
        
        uniprot_id = self.kw[index]["uniprot_id"]
        answer = self.kw[index]["A"]
        answer_id = self.answer_map[answer]
       
        seq = self.sequence[uniprot_id]

        if len(seq) > 600:
            seq = seq[:600]

        return {
            "seq": seq,
            "answer_id": answer_id
        }

    # stage1-Qformer
        # uniprot_id = self.annotation[index]["uniprot_id"]
        # seq = self.sequence[uniprot_id]
        # answer = self.annotation[index]["name"]

        # if len(seq) > 1024:
        #     seq = seq[:1024]

        # return {
        #     "seq": seq,
        #     "text_input": answer
        # }

class SeqEvalDataset(BaseDataset):
    def __init__(self, kw_path, text_rule_path, text_manual_path, seq_path):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # print("______Enter Seq Dataset____")
        # super().__init__(vis_processor, text_processor)
        # self.qa_path = qa_path
        # self.seq_path = seq_path

        self.kw = json.load(open(kw_path, "r")) 
        self.rule = json.load(open(text_rule_path, "r"))
        self.manual = json.load(open(text_manual_path, "r"))
        self.sequence = json.load(open(seq_path, "r"))

        self.rate = {'kw':1, 'rule':1, 'manual':1}
        self.len_kw = len(self.kw)
        self.len_rule = len(self.rule)
        self.len_manual = len(self.manual)

        self.split1 = self.rate['kw'] * self.len_kw 
        self.split2 = self.split1 + self.rate['rule'] * self.len_rule
        self.split3 = self.split2 + self.rate['manual'] * self.len_manual 

        # print(self.len_kw, self.len_rule, self.len_manual)
        # print(self.split1, self.split2, self.split3)

    def __len__(self):
        return self.split3

    def __getitem__(self, index):
        
        if index < self.split1: # sample kw 
            uniprot_id = self.kw[index]["uniprot_id"]
            answer = self.kw[index]["A"]
            prompt = f"###Human: <protein><proteinHere></protein> {self.kw[index]['Q']} ###Assistant:"
            eval_type = 0
        elif index < self.split2: # sample rule based functionality
            true_index  = (index - self.split1) % self.len_rule
            uniprot_id = self.rule[true_index]["uniprot_id"]
            answer = self.rule[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
            eval_type = 1
        else: # sample manual annotated functionality
            true_index  = (index - self.split2) % self.len_manual
            uniprot_id = self.manual[true_index]["uniprot_id"]
            answer = self.manual[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
            eval_type = 2
        
        seq = self.sequence[uniprot_id]

        if len(seq) > 600:
            seq = seq[:600]

        return {
            "seq": seq,
            "text_input": answer,
            "prompt": prompt,
            "eval_type": eval_type
        }

    
