import os
import sys
from proteinchat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random

questions = ["Tell me about this protein.", 
                "What is the functionality of this protein?", 
                "Briefly summarize the functionality of this protein.",
                "Please provide a detailed description of the protein."]
q_map = {
    "Can this protein bind to RNA?":
    " Reply only with Yes or No.",
    "Can this protein bind to DNA?":
    " Reply only with Yes or No.",
    "What type of enzyme is this?":
    " Choose one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
    "What type of protein is this?":
    " Choose one from Ribonucleoprotein and Chaperone protein",
    "What electron acceptor or cofactor does this enzyme use?":
    " Choose one from NAD and NADP.",
    "What ligand can this protein bind to?":
    " Choose one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
    "Which cellular or extracellular component can this protein be found in?":
    " Choose one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
    "What biological process does this protein involved in?":
    " Choose one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
}
class SeqDataset(BaseDataset):
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

        # self.kw = random.sample(self.kw, 100000)
        # self.rule = random.sample(self.rule, 100000)

        self.rate = {'kw':1, 'rule':1, 'manual':4}
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
            query = self.kw[index]['Q']
            query += q_map[query]
            prompt = f"###Human: <protein><proteinHere></protein> {query} ###Assistant:"
        elif index < self.split2: # sample rule based functionality
            true_index  = (index - self.split1) % self.len_rule
            uniprot_id = self.rule[true_index]["uniprot_id"]
            answer = self.rule[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        else: # sample manual annotated functionality
            true_index  = (index - self.split2) % self.len_manual
            uniprot_id = self.manual[true_index]["uniprot_id"]
            answer = self.manual[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        
        seq = self.sequence[uniprot_id]

        if len(seq) > 600:
            seq = seq[:600]

        return {
            "seq": seq,
            "text_input": answer,
            "prompt": prompt
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


