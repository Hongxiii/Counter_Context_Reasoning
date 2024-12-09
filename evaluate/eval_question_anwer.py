import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import argparse
import json 
tokenizer = BertTokenizer.from_pretrained('../models/bert-base-cased')
model = BertModel.from_pretrained('../models/bert-base-cased')


def calculate_bem(candidate, reference):
    # Tokenize input
    inputs = tokenizer([candidate, reference], return_tensors='pt', padding=True, truncation=True)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return similarity[0][0]


def calculate_wups(candidate, reference, threshold):
    def get_synsets(word):
        return wn.synsets(word)

    def max_similarity(synsets1, synsets2):
        max_sim = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                sim = wn.wup_similarity(syn1, syn2) or 0.0
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    # Tokenize and get synsets
    answer_tokens = candidate.split()
    reference_tokens = reference.split()

    total_score = 0.0
    for a in answer_tokens:
        max_sim = max(max_similarity(get_synsets(a), get_synsets(r)) for r in reference_tokens)
        total_score += max_sim if max_sim > threshold else 0.0

    return total_score / len(answer_tokens) 


PATH = {
    'BLIP-Base': '../results/question_answer/blip-vqa-base.json',
    'BLIP2-XL': '../results/question_answer/blip2-flan-t5-xl.json',
    'BLIP2-XXL': '../results/question_answer/blip2-flan-t5-xxl.json',
    'InstructBLIP-XL': '../results/question_answer/instructblip-flan-t5-xl.json',
    'InstructBLIP-XXL': '../results/question_answer/instructblip-flan-t5-xl.json',
    'mPLUG-owl-7B': '../results/question_answer/mplug-owl-llama-7b.json',
    'mPLUG-owl2-7B': '../results/question_answer/mplug-owl2-llama2-7b.json',
    'LLaVA-1.5-7B': '../results/question_answer/llava-v1.5-7b.json',
    'LLaVA-1.6-7B': '../results/question_answer/llava-v1.6-vicuna-7b.json',
    'MMICL': '../results/question_answer/MMICL-Instructblip-T5-xl.json',
    'OpenFlamingo': '../results/question_answer/OpenFlamingo-3B-vitl-mpt1b.json',
    'GPT-4V': '../results/question_answer/sk-hkR7ohdkEqFjD9MiEc22B0C392484eE692Ca0aD3F8B06c5c.json'
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True, choices=list(PATH.keys()))
    args = parser.parse_args()
    # 模型路径
    true_path = "../dataset/data_vqa.json"
    file_path = PATH[args.path]
    BEMs = []
    WUPS_0s = []
    WUPS_9s = []

    with open(file_path, 'r') as p_file:
        pre_data = json.load(p_file)

    with open(true_path, 'r') as t_file:
        true_data = json.load(t_file)

    for key in tqdm(pre_data.keys()):
        true_dict = {}
        pre_dict = {}
        for t_pair, p_pair in zip(true_data[key], pre_data[key]):
            true_dict[t_pair["question"]] = t_pair["answer"]
            pre_dict[p_pair["question"]] = p_pair["answer"]
        
        for question, answer in pre_dict.items():
            crowded_capton = true_dict[question]
            predicted_answer = pre_dict[question]
            # print(key)
            # print(question)
            # print(crowded_capton)
            # print(predicted_answer)
            BEM = calculate_bem(predicted_answer, crowded_capton)
            try:
                wups_0 = calculate_wups(predicted_answer, crowded_capton, threshold=0.0)
                wups_9 = calculate_wups(predicted_answer, crowded_capton, threshold=0.9)
            except: 
                print(key)
                print(crowded_capton)
                print(predicted_answer)
                print(question)
            BEMs.append(BEM)
            WUPS_0s.append(wups_0)
            WUPS_9s.append(wups_9)

print(len(BEMs))
print(f"BEM mean:{sum(BEMs)/len(BEMs)}")
print(f"WUPS@0.0 mean:{sum(WUPS_0s)/len(WUPS_0s)}")
print(f"WUPS@0.9 mean:{sum(WUPS_9s)/len(WUPS_9s)}")



