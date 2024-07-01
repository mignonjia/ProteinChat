import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import json
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score

# Import our models. The package will take care of downloading the models automatically

def get_simcse(simcse_path, func_text):
    tokenizer = AutoTokenizer.from_pretrained(simcse_path)
    model = AutoModel.from_pretrained(simcse_path)
    
    refs = [item['correct_func'] for item in func_text]
    # refs = [item if len(item) < 1000 else item[:1000] for item in refs]
    cands = [item['predict_func'] for item in func_text]
    # cands = [item if len(item) < 1000 else item[:1000] for item in cands]

    ref_tokens = tokenizer(refs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref_embeddings = model(**ref_tokens, output_hidden_states=True, return_dict=True).pooler_output

    cand_tokens = tokenizer(cands, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        cand_embeddings = model(**cand_tokens, output_hidden_states=True, return_dict=True).pooler_output

    weights = [(1.,0,0,0), (1./2., 1./2., 0, 0), (1./3., 1./3., 1./3., 0), (1./4., 1./4., 1./4., 1./4.)]

    # bleu_list, meteor_list, simcse_list = [[] for i in range(4)], [], []
    bleu_list, simcse_list = [[] for i in range(4)], []
    for i in range(len(func_text)):
        item = func_text[i]
        correct = item['correct_func'].split()
        predict = item['predict_func'].split()
        bleu = sentence_bleu([correct], predict, weights)
        # meteor = meteor_score([correct], predict)
        simcse = 1 - cosine(ref_embeddings[i], cand_embeddings[i])
        func_text[i]['simcse'] = simcse
        func_text[i]['bleu'] = bleu
        # func_text[i]['meteor'] = meteor
        for ngram in range(4):
            bleu_list[ngram].append(bleu[ngram])
        # meteor_list.append(meteor)
        simcse_list.append(simcse)

    scores = {}
    # for item in score_list:
    #     scores.append(item)
    #     val_list.append(item['eval'])

    for ngram in range(4):
        scores[f'average_bleu_{str(ngram+1)}'] = sum(bleu_list[ngram]) / len(bleu_list[ngram])
        
    # scores['average_meteor'] = sum(meteor_list) / len(meteor_list)
    scores['average_simcse'] = sum(simcse_list) / len(simcse_list)
    print("Average scores:")
    print(scores)

    func_text.append(scores)

    return func_text
    # before finetune: score-manual 0.6840915057589027, score rule-based 0.7981250693743256
    # after finetune: score-manual 0.6872740607051289, score rule-based 0.7505798617313648
    

def get_simcse_llm_param(simcse_path, func_text):
    tokenizer = AutoTokenizer.from_pretrained(simcse_path)
    model = AutoModel.from_pretrained(simcse_path)
    
    refs = [item['correct_func'] for item in func_text]
    # refs = [item if len(item) < 1000 else item[:1000] for item in refs]
    cands = [item['predict_func'] for item in func_text]
    # cands = [item if len(item) < 1000 else item[:1000] for item in cands]

    ref_tokens = tokenizer(refs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref_embeddings = model(**ref_tokens, output_hidden_states=True, return_dict=True).pooler_output

    cand_tokens = tokenizer(cands, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        cand_embeddings = model(**cand_tokens, output_hidden_states=True, return_dict=True).pooler_output

    weights = [(1.,0,0,0), (1./2., 1./2., 0, 0), (1./3., 1./3., 1./3., 0), (1./4., 1./4., 1./4., 1./4.)]

    bleu_list = {}
    # meteor_list = [[[] for i in range(2)] for i in range(3)]
    simcse_list = {}

    i = 0
    while i < len(func_text):
        item = func_text[i]
        correct = item['correct_func'].split()
        predict = item['predict_func'].split()
        bleu = sentence_bleu([correct], predict, weights)
        # meteor = meteor_score([correct], predict)
        simcse = 1 - cosine(ref_embeddings[i], cand_embeddings[i])
        func_text[i]['simcse'] = simcse
        func_text[i]['bleu'] = bleu
        # func_text[i]['meteor'] = meteor

        num_beams = func_text[i]['num_beams']
        temperature = func_text[i]['temperature']

        if num_beams not in bleu_list:
            bleu_list[num_beams] = {}
            simcse_list[num_beams] = {}
        
        if temperature not in bleu_list[num_beams]:
            bleu_list[num_beams][temperature] = [[] for i in range(4)]
            simcse_list[num_beams][temperature] = []

        for ngram in range(4):
            bleu_list[num_beams][temperature][ngram].append(bleu[ngram])
        # meteor_list[num_beam][temp].append(meteor)
        simcse_list[num_beams][temperature].append(simcse)
        i += 1

    # for item in score_list:
    #     scores.append(item)
    #     val_list.append(item['eval'])
    print("bleu_list", bleu_list)
    print("simcse_list", simcse_list)
    scores = {}
    for num_beams, val in bleu_list.items():
        scores[num_beams] = {}
        for temperature, _ in val.items():
            scores[num_beams][temperature] = {}
            for ngram in range(4):
                scores[num_beams][temperature][f'average_bleu_{str(ngram+1)}'] = round(sum(bleu_list[num_beams][temperature][ngram]) / len(bleu_list[num_beams][temperature][ngram]), 2)
        
            # scores['average_meteor'] = round(sum(meteor_list[num_beam][temp]) / len(meteor_list[num_beam][temp]), 2)
            scores[num_beams][temperature]['average_simcse'] = round(sum(simcse_list[num_beams][temperature]) / len(simcse_list[num_beams][temperature]), 2)
                
    print("scores:")
    print(scores)
    func_text.append(scores)

    return func_text
    # before finetune: score-manual 0.6840915057589027, score rule-based 0.7981250693743256
    # after finetune: score-manual 0.6872740607051289, score rule-based 0.7505798617313648
 

if  __name__ == "__main__":
    #get_correct_func_words()
    simcse_path = "/nfs_baoding_ai/mingjia_2023/sup-simcse-roberta-large"
    out_dir = "0-init/simcse/"

    with open("0-init/unstructured-train.json", 'r') as f:
        func_text = json.load(f)
    
    get_simcse(simcse_path, func_text, out_dir)