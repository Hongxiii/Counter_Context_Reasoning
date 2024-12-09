import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bleurt import score
import statistics

import argparse
import json
import pandas as pd
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_excel(file_path, key_head):
    df = pd.read_excel(file_path,sheet_name=0)

    id_captions_dict = {}
    for index, row in df.iterrows():
        id = row['ID']
        if id.lower().startswith(key_head):
            captions = [
                row['Crowd Captions[0]'], 
                row['Crowd Captions[1]'], 
                row['Crowd Captions[2]'], 
                row['Crowd Captions[3]'], 
                row['Crowd Captions[4]']
            ]
            captions = [caption for caption in captions if pd.notna(caption)]
            id_captions_dict[id] = captions

    return id_captions_dict


def calculate_bleu4(candidate, references):
    smoothie = SmoothingFunction().method4
    bleu4_score = sentence_bleu([nltk.word_tokenize(ref) for ref in references], nltk.word_tokenize(candidate),
                                smoothing_function=smoothie)
    return bleu4_score


def calculate_rougel(candidate, references):
    rouge_scorer = Rouge()
    rouge_l_score = statistics.mean([rouge_scorer.get_scores(candidate, ref)[0]['rouge-l']['f'] for ref in references])
    return rouge_l_score


def calculate_cider(candidate, references):
    scorer = Cider()
    (score, scores) = scorer.compute_score(references, candidate)
    return score


def calculate_bleurt(candidate, references):
    bleurt_scorer = score.BleurtScorer('../models/BLEURT-20')
    bleurt_score = statistics.mean(bleurt_scorer.score(references=references, candidates=[candidate] * len(references)))
    return bleurt_score


PATH = {
    # 'BLIP-Base': '../results/image_caption/blip-image-captioning-base',
    # 'BLIP2-XL': '../results/image_caption/blip2-flan-t5-xl',
    # 'BLIP2-XXL': '../results/image_caption/blip2-flan-t5-xxl',
    # 'InstructBLIP-XL': '../results/image_caption/instructblip-flan-t5-xl',
    # 'InstructBLIP-XXL': '../results/image_caption/instructblip-flan-t5-xxl',
    # 'mPLUG-owl-7B': '../results/image_caption/mplug-owl-llama-7b',
    # 'mPLUG-owl2-7B': '../results/image_caption/mplug-owl2-llama2-7b',
    # 'LLaVA-1.5-7B': '../results/image_caption/llava-v1.5-7b',
    # 'LLaVA-1.6-7B': '../results/image_caption/llava-v1.6-vicuna-7b',
    # 'MMICL': '../results/image_caption/MMICL-Instructblip-T5-xl',
    'OpenFlamingo': '../results/image_caption/OpenFlamingo-3B-vitl-mpt1b',
    'GPT-4V':'../results/image_caption/sk-hkR7ohdkEqFjD9MiEc22B0C392484eE692Ca0aD3F8B06c5c'
}

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Evaluate predicted captions against crowd captions.')
    # parser.add_argument("-model", type=str, required=True, choices=list(PATH.keys()))
    # args = parser.parse_args()

    for model in PATH.keys(): 
        print(model)
        json_file_path = PATH[model]+'.json'
        excel_file_path = '../dataset/annotation.xlsx'

        key_heads = ["a", "b", "c", "d", "e"] 
        class_res = {}
        total_bleu4 = []
        total_rougel = []
        total_bleurt = []
        total_cider = []
        for key_head in key_heads:

            predictions = load_json(json_file_path)
            df = pd.read_excel(excel_file_path)
            # print(df)

            bleu4_scores = []
            rougel_scores = []
            bleurt_scores = []

            id_to_row = {row['ID']: row for _, row in df.iterrows() if row['ID'].lower().startswith(key_head)}
            predictions = {key:value for key, value in predictions.items() if key.lower().startswith(key_head)}

            # 由于部分图片缺少，所以预测的数量会比标注的数量少一些

            for prediction_id, predicted_caption in tqdm(predictions.items(), desc="Processing"):
                if prediction_id in id_to_row:
                    row = id_to_row[prediction_id]
                    crowded_captions = [row[f'Crowd Captions[{i}]'] for i in range(5) if pd.notna(row[f'Crowd Captions[{i}]'])]
                    
                    bleu4_score = calculate_bleu4(predicted_caption, crowded_captions)
                    rougel_score = calculate_rougel(predicted_caption, crowded_captions)
                    bleurt_score = calculate_bleurt(predicted_caption, crowded_captions)
                    
                    total_bleu4.append(bleu4_score)
                    total_rougel.append(rougel_score)
                    total_bleurt.append(bleurt_score)

                    bleu4_scores.append(bleu4_score)
                    rougel_scores.append(rougel_score)
                    bleurt_scores.append(bleurt_score)
                

            avg_bleu4 = statistics.mean(bleu4_scores) if bleu4_scores else 0
            avg_rougel = statistics.mean(rougel_scores) if rougel_scores else 0
            avg_bleurt = statistics.mean(bleurt_scores) if bleurt_scores else 0


            candidate = {k: [v] for k, v in predictions.items()}
            references = load_excel(excel_file_path, key_head)
            # 由于部分图片缺少，所以预测的数量会比标注的数量少一些，在这里取交集
            common_keys = set(candidate.keys()).intersection(references.keys())
            candidate = {k: candidate[k] for k in common_keys}
            references = {k: references[k] for k in common_keys}

            avg_cider = calculate_cider(candidate, references)
            total_cider.append(avg_cider)
            # print(avg_cider)

            class_res[key_head] = {
                "Average BLEU-4 score": avg_bleu4,
                "Average ROUGE-L score": avg_rougel,
                "Average CIDEr score": avg_cider,
                "Average BLEURT score": avg_bleurt
            }

        class_res['total'] = {
                "Average BLEU-4 score": statistics.mean(total_bleu4),
                "Average ROUGE-L score": statistics.mean(total_rougel),
                "Average CIDEr score": statistics.mean(total_cider),
                "Average BLEURT score": statistics.mean(total_bleurt),
        }
        
        with open(f"./image_caption/{model}"+".json", 'w') as outfile:
            json.dump(class_res, outfile, indent=4)
