import argparse
import os
import random
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from collections import Counter

from proteinchat.common.config import Config
from proteinchat.common.registry import registry
from proteinchat.common.dist_utils import get_rank, init_distributed_mode
from proteinchat.common.conversation import Chat, CONV_VISION

from eval import get_simcse, get_simcse_llm_param
import json

# Imports PIL module
from PIL import Image

# imports modules for registration
from proteinchat.datasets.builders import *
from proteinchat.models import *
from proteinchat.runners import *
from proteinchat.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",
                        default='configs/proteinchat_eval.yaml')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
init_distributed_mode(cfg.run_cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_protein(seq):
    chat_state = CONV_VISION.copy()
    img_list = []
    protein_emb, llm_message = chat.upload_protein(seq, chat_state, img_list)
    return chat_state, img_list, protein_emb

def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams=1, temperature=1e-3, top_p = 0.9, save_embeds=False):
    # print(chat_state)
    llm_message, _, loss = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              top_p = top_p,
                              #repetition_penalty=2.0,
                              max_new_tokens=200,
                              max_length=1500, 
                              save_embeds=save_embeds)
    return llm_message, chat_state, img_list, loss

def gradio_ppl(chat_state, img_list, predict_list):
    # print(chat_state)
    loss = chat.get_ppl(conv=chat_state,
                              img_list=img_list,
                              predict_list=predict_list)
    return loss


old_questions = ["Tell me about this protein in one or two sentences.", 
                "What is the functionality of this protein? Reply within one or two sentences", 
                "Briefly summarize the functionality of this protein in one or two sentences.",
                "Please provide a description of the protein in one or two sentences."]
questions = ["Tell me about this protein.", 
                "What is the functionality of this protein?", 
                "Briefly summarize the functionality of this protein.",
                "Please provide a detailed description of the protein."]

def eval_ppl():
    func_text = []
    qa_list = json.load(open(f"/nfs_beijing_ai/mingjia_2023/proteinchat_glm/results-glm/10-glm-scratch-llama2-kw/params/ckpt3_beam4_joint_list.json"))
    loss_list = []

    for item in qa_list:
        predict_list = item['predict_func']

        seq = item['seq']
        query = item['query']

        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        loss = gradio_ppl(chat_state, img_list, predict_list)

        item['loss'] = loss
        func_text.append(item)
        print(predict_list)
        print("loss", loss)
        print('='*80)
    with open("/nfs_beijing_ai/mingjia_2023/proteinchat_glm/results-glm/10-glm-scratch-llama2-kw/confidence/ckpt3_beam4_joint.json", "w") as outfile:
        json.dump(func_text, outfile, indent=4)
    return func_text

def eval_antimicro():
    for split in ['manual', 'rule']:
        qa_list = json.load(open(f"/nfs_beijing_ai/mingjia_2023/data/antimicro/{split}.json"))
        func_text = []
        print('Num_beam: 4')
        print('max_new_tokens: 200, all previous tests: 100')
        print('temperature: 1e-3')
        print('top_p: 0.9')
        for item in qa_list:
            uniprot_id = item['uniprot_id']
            seq = item['seq']
            query = random.choice(questions)
            if len(seq) > 600:
                seq = seq[:600]

            user_message = query
            chat_state, img_list, protein_embs = upload_protein(seq)
            chat_state = gradio_ask(user_message, chat_state)

            llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=4)

            entry = {"seq": seq, "query": query, "correct_func": item['correct_func'], "predict_func": llm_message}
            func_text.append(entry)
            print("Uniprot ID:", uniprot_id)
            print(f"Predicted Function: {llm_message}")
            print('='*80)

        simcse_path = "../sup-simcse-roberta-large"
        scores = get_simcse(simcse_path, func_text)

        with open(f"/nfs_beijing_ai/mingjia_2023/data/antimicro/{split}_gen.json", "w") as outfile:
            json.dump(scores, outfile, indent=4)
    
def eval_func_text(qa_list, seq):
    start = time.time()
    func_text = []
    loss_list = []

    for item in qa_list:

        function = item['caption']
        uniprot_id = item['uniprot_id']
        seq = seqs[uniprot_id]
        query = random.choice(questions)

        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list)

        loss_list.append(loss)
        entry = {"seq": seq, "query": query, "correct_func": function, "predict_func": llm_message}
        func_text.append(entry)

        print("Uniprot ID:", uniprot_id)
        print("Loss:", loss)
        print("Correct Function:", function)
        print(f"Predicted Function: {llm_message}")
        print('='*80)

    ppl = math.exp(sum(loss_list)/len(loss_list))
    print(ppl)
    end = time.time()
    print(end - start)
    print("******************")

    return func_text

def eval_multi_round():
    func_text = []
    file_path = "data/multi_round/sample.json"
    qa_list = json.load(open(file_path))
    loss_list = []

    for item in qa_list:

        function = item['correct_func']
        seq = item['seq']
        query = item['query']

        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=4)
        # message_2 = "What specific antibacterial activity?"
        # message_2 = "Can you elaborate on the specific type of histone protein described, its unique properties, and its function in the regulation of DNA accessibility within cells?"
        message_2 = "What ligand can this protein bind to?"
        chat_state = gradio_ask(message_2, chat_state)

        llm_message_2, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=1)

        message_3 = "Which metal is this protein capable of binding?"
        chat_state = gradio_ask(message_3, chat_state)

        llm_message_3, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=1)

        loss_list.append(loss)
        entry = {"seq": seq, "query": query, "correct_func": function, "predict_func_1": llm_message, "query_2": message_2, "predict_func_2": llm_message_2, "query_3": message_3, "predict_func_3": llm_message_3}
        func_text.append(entry)

        print("seq:", seq)
        print("Correct Function:", function)
        print(f"Predicted Function 1: {llm_message}")
        print(f"Predicted Function 2: {llm_message_2}")
        print('='*80)

    with open("../data/multi_round/result_3.json", "w") as outfile:
        json.dump(func_text, outfile, indent=4)
    return func_text


def eval_LLM_params(qa_list, seq):
    start = time.time()
    func_text = []
    num_beams_list = [1, 2, 4, 8]
    # temperature_list = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    l = len(num_beams_list)

    loss_list = [[] for i in range(l)] 
    ppl_list = [0 for i in range(l)] 
    # random_numbers = random.sample(list(range(len(qa_list))), k=NUM_TEST)

    for item in qa_list:
        # item = qa_list[random_numbers[i]]

        function = item['caption']
        uniprot_id = item['uniprot_id']
        seq = seqs[uniprot_id]
        query = random.choice(questions)

        if len(seq) > 600:
            seq = seq[:600]

         # top_p: 0.9, 0.99 no difference 
        for i in range(l):
            # num_beams = 1
            # temperature = temperature_list[i]
            num_beams = num_beams_list[i]
            temperature = 1e-3

            user_message = query
            chat_state, img_list, protein_embs = upload_protein(seq)
            chat_state = gradio_ask(user_message, chat_state)

            llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=num_beams, temperature=temperature)
            loss_list[i].append(loss)

            entry = {"uniprot_id": uniprot_id, "seq": seq, "query": query, "correct_func": function, "predict_func": llm_message, "num_beams": num_beams, "temperature": temperature, "ppl": loss}
            func_text.append(entry)

            print("Uniprot ID:", uniprot_id)
            print("Query:", query)
            print("Correct Function:", function)
            print(f"Predicted Function: {llm_message}")
    
    print(loss_list)
    for i in range(l):
        ppl_list[i] = sum(loss_list[i])/len(loss_list[i])

    print(ppl_list)

    end = time.time()
    print(end - start)
    print("******************")

    return func_text


q_map = {
    "Can this protein bind to RNA?":
    " Reply only with Yes or No.",
    "Can this protein bind to DNA?":
    " Reply only with Yes or No.",
    "What type of enzyme is this?":
    " Choose only one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
    "What type of protein is this?":
    " Choose only one from Ribonucleoprotein and Chaperone protein",
    "What electron acceptor or cofactor does this enzyme use?":
    " Choose only one from NAD and NADP.",
    "What ligand can this protein bind to?":
    # " Choose one from metals.",
    " Choose only one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
    "Which cellular or extracellular component can this protein be found in?":
    " Choose only one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
    "What biological process does this protein involved in?":
    " Choose only one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
}

def eval_kw(qa_list, seqs):
    start = time.time()
    # print("******************")
    # qa_file = "/nfs_beijing/kubeflow-user/mingjia_2023/data/valid_set/qa.json"
    
    func_text = []
    acc_cnt = Counter()
    sample_cnt = Counter()

    for item in qa_list:
        function = item['A']
        if ',' in function:
            continue
        if item['Q_id'] >= 6:
            continue
        uniprot_id = item['uniprot_id']
        query = item['Q']
        query += q_map[query]

        seq = seqs[uniprot_id]
        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list)

        sample_cnt[query] += 1
        if function.strip().lower() in llm_message.strip().lower():
            acc_cnt[query] += 1
        item['predict_func'] = llm_message
        func_text.append(item)
        print(item)

    acc = []  
    for key, value in acc_cnt.items():
        func_text.append({key: [value, sample_cnt[key], value/sample_cnt[key]]})
        print(func_text[-1])
        acc.append(value/sample_cnt[key])

    print(sum(acc)/len(acc))
    func_text.append(sum(acc)/len(acc))

    end = time.time()
    print(end - start)
    print("******************")

    return func_text

def tsne_one_seq(function, seq):

    if len(seq) > 600:
        seq = seq[:600]
    
    query_list = ["Co-chaperone involved in the maturation of iron-sulfur cluster-containing proteins. Seems to help targeting proteins to be folded toward HscA", random.choice(questions)]

    for query in query_list:
        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        print(protein_embs.squeeze().detach().cpu().numpy().shape)
        np.save('/nfs_beijing_ai/mingjia_2023/proteinchat_glm/tsne/protein.npy', protein_embs.squeeze().detach().cpu().numpy())

        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, save_embeds=True)

        entry = {"query": query, "correct_func": function, "predict_func": llm_message}
        func_text.append(entry)

        print("Query:", query)
        print("Correct Function:", function)
        print(f"Predicted Function: {llm_message}")
   
    return func_text

def tsne_multi_seq(prots):
    
    encoding_array = []
    for entry in prots:
        uniprot_id = entry['uniprot_id']
        seq = entry['seq']
        tag = entry['Class']

        if len(seq) > 600:
            seq = seq[:600]

        chat_state, img_list, protein_embs = upload_protein(seq)
        protein_embs = torch.mean(protein_embs, 1)
        
        encoding_array.append(protein_embs.squeeze().detach().cpu().numpy())
    
    encoding_array = np.array(encoding_array)   
    print(encoding_array.shape)
    np.save('tsne/protein.npy', encoding_array)


if  __name__ == "__main__":
    # eval_ppl()

    # eval_multi_round()

    # result_dir = "results-glm/10-glm-scratch-llama2-kw/ckpt3"

    # for data_dir in ['test']: #'train', 
    #     seqs = json.load(open(f"data/{data_dir}_set/seq.json"))
    #     qa_list = json.load(open(f"data/{data_dir}_set/before_combine/subset/qa_kw.json"))
    #     scores = eval_kw(qa_list, seqs)
    #     with open("tmp.json", "w") as outfile:
    #         json.dump(scores, outfile, indent=4)

    # eval func text & kw
    for data_dir in ['valid']: #'train', 
        seqs = json.load(open(f"data/{data_dir}_set/seq.json"))

        for qa_file in ['manual', 'rule']:
            qa_list = json.load(open(f"data/{data_dir}_set/subset/qa_text_{qa_file}.json"))
            func_text = eval_func_text(qa_list, seqs)

            # with open(f"results/esm.json", "a") as outfile:
            #     json.dump(func_text, outfile, indent=4)
            
            simcse_path = "../sup-simcse-roberta-large"
            print(data_dir, qa_file)

            scores = get_simcse_llm_param(simcse_path, func_text)
            get_simcse(simcse_path, func_text)

            with open("results/esm.json", "a") as outfile:
                json.dump(scores, outfile, indent=4)

        qa_list = json.load(open(f"data/{data_dir}_set/subset/qa_kw.json"))
        scores = eval_kw(qa_list, seqs)
        with open("results/esm.json", "a") as outfile:
            json.dump(scores, outfile, indent=4)


