import argparse
import sys
import os
import pandas as pd
import re
import random
import torch
import json
from tqdm import tqdm

def load_dataset(dataset_path, ispre):
    is_predict = True if ispre=="yes" else False
    # 读取图像路径
    images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.webp', '.jpg')):
                image_path = os.path.join(root, file)
                images.append(image_path)
    
    file_path = '../results/image_caption/blip2-flan-t5-xl.json'
# 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        predict_captions = json.load(file)
    # 读取图像描述和解释
    prompts = []
    annotation_file = dataset_path + "/dataset.xlsx"
    annotation_data = pd.ExcelFile(annotation_file)
    true_data = annotation_data.parse(annotation_data.sheet_names[1])  # 正样本子表
    false_data = annotation_data.parse(annotation_data.sheet_names[2])  # 负样本子表
    true_caption = true_data.iloc[:, 0:6].set_index(true_data.columns[0])[true_data.columns[5]].to_dict()  # 正样本描述
    false_caption = false_data.iloc[:, 0:6].set_index(false_data.columns[0])[false_data.columns[5]].to_dict()  # 负样本描述
    true_explanation = true_data.iloc[:, 0:7].set_index(true_data.columns[0])[true_data.columns[6]].to_dict()  # 正样本解释
    false_explanation = false_data.iloc[:, 0:7].set_index(false_data.columns[0])[false_data.columns[6]].to_dict()  # 负样本解释
    caption_data = true_caption | false_caption
    explanation_data  = true_explanation | false_explanation
    # 统一列表顺序
    captions, explanations = [], []
    for i in images:
        image_id = os.path.splitext(os.path.basename(i))[0]
        background = i.split('/')[-2]
        if is_predict:
            if image_id in predict_captions.keys():
                captions.append(predict_captions[image_id])
                # print(predict_captions[image_id])
            else:
                continue
        else:
            captions.append(re.findall(r'\[(.*?)\]', caption_data[image_id])[0])  # 取第一条标注描述
        if image_id.split('-')[1] == 'f':
            explanations.append(re.findall(r'\[(.*?)\]', explanation_data[image_id])[0])  # 取第一条标注解释
        else:
            explanations.append(f'This image is consistent with the background knowledge of \"{background}\".')
    return images, captions, explanations



def llama(model_path, images, captions, explanations, setting, withCoT, ispre):
    # 模型加载及配置
    from llama import Llama
    import ollama

    # generator = Llama.build(
    #     ckpt_dir=model_path,
    #     tokenizer_path=model_path+"/tokenizer.model",
    #     max_seq_len=512,
    #     max_batch_size=1,
    #     model_parallel_size=1
    # )
    # 零样本设定提示词
    if setting == 'z':
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''This is a caption of an image: {c} \n Explain why this description doesn't match the background knowledge of \"{background}\"? Use a paragraph to explain!'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough,\" and provide supportingevidence for your answer.\nUSER: {c} Does this description match the background knowledge of \"{background}\"? Let's think step by step.\nASSISTANT:'''
                prompts.append(prompt)
    # 少样本设定提示词（2个样本）
    else:
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                answer_1 = 'Yes' if sample_1.split('/')[-1][2] == 't' else 'No'  # 样本答案
                answer_2 = 'Yes' if sample_2.split('/')[-1][2] == 't' else 'No'
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough\".\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{answer_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{answer_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                explanation_1, explanation_2 = explanations[index_1], explanations[index_2]  # 样本解释
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough,\" and provide supportingevidence for your answer.\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{explanation_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{explanation_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
    # 模型推理
    result = {}
    for image, prompt in tqdm(zip(images, prompts)):
        image_id = image.split('/')[-1].split('.')[0]

        if "f" in image_id:

            generation_output = ollama.chat(model="llama2",stream=False,messages=[{"role": "user","content": f"{prompt}"}])

            output = generation_output["message"]['content']
            
            # print(image_id, output)
            result[image_id] = output
        

    # 结果保存
    with open("../results/pipeline_explanation/"+model_path.split('/')[-1]+'-'+setting+'-'+withCoT+'-'+ispre+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def vicuna(model_path, images, captions, explanations, setting, withCoT, ispre):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from FastChat.fastchat.model.model_adapter import (
        load_model,
        get_conversation_template,
        get_generate_stream_function,
    )
    from FastChat.fastchat.modules.awq import AWQConfig
    from FastChat.fastchat.modules.gptq import GptqConfig
    from FastChat.fastchat.utils import get_context_length
    from FastChat.fastchat.serve.cli import SimpleChatIO
    # 模型加载及配置
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        dtype=None,
        load_8bit=False,
        cpu_offloading=False,
        gptq_config=GptqConfig(
            ckpt=model_path,
            wbits=16,
            groupsize=-1,
            act_order=False,
        ),
        awq_config=AWQConfig(
            ckpt=model_path,
            wbits=16,
            groupsize=-1,
        ),
        exllama_config=None,
        xft_config=None,
        revision="main",
        debug=False,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)
    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    is_xft = "xft" in model_type
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2
    context_len = get_context_length(model.config)

    # 零样本设定提示词
    if setting == 'z':
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''This is a caption of an image: {c} \n Explain why this description doesn't match the background knowledge of \"{background}\"? Use a paragraph to explain! \nASSISTANT:'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough,\" and provide supportingevidence for your answer.\nUSER: {c} Does this description match the background knowledge of \"{background}\"? Let's think step by step.\nASSISTANT:'''
                prompts.append(prompt)
    # 少样本设定提示词（2个样本）
    else:
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                answer_1 = 'Yes' if sample_1.split('/')[-1][2] == 't' else 'No'  # 样本答案
                answer_2 = 'Yes' if sample_2.split('/')[-1][2] == 't' else 'No'
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough\".\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{answer_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{answer_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                explanation_1, explanation_2 = explanations[index_1], explanations[index_2]  # 样本解释
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough,\" and provide supportingevidence for your answer.\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{explanation_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{explanation_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
    # 模型推理
    result = {}
    for image, prompt in tqdm(zip(images, prompts)):
        image_id = image.split('/')[-1].split('.')[0]
        if "f" in image_id:
            gen_params = {
                "model": model_path,
                "prompt": prompt,
                "temperature": 1,
                "repetition_penalty": 1,
                "max_new_tokens": 512,
                "stop": None,
                "stop_token_ids": None,
                "echo": False,
            }
            chatio = SimpleChatIO(False)
            chatio.prompt_for_output('USER')
            chatio.prompt_for_output('ASSISTANT')
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                "cuda",
                context_len=context_len,
                judge_sent_end=False,
            )
            output = chatio.stream_output(output_stream)
            
            print(image_id, output)
            result[image_id] = output

    
    # 结果保存
    with open("../results/pipeline_explanation/"+model_path.split('/')[-1]+'-'+setting+'-'+withCoT+'-'+ispre+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def gpt(model_path, images, captions, explanations, setting, withCoT, ispre):
    # 模型加载及配置
    import httpx
    from openai import OpenAI
    client = OpenAI(
        base_url="https://oneapi.xty.app/v1", 
        api_key=model_path,
        http_client=httpx.Client(
            base_url="https://oneapi.xty.app/v1",
            follow_redirects=True,
        ),
    )

    # 零样本设定提示词
    if setting == 'z':
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''This is a caption of an image: {c} \n Explain why this description doesn't match the background knowledge of \"{background}\"? Use a paragraph to explain! DO NOT ASK "what is the background?"\n ASSISTANT:'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background = i.split('/')[-2]
                prompt = f'''You will be presented with a question.You should answer \"Yes\" or \"No\" and provide supportingevidence for your answer.\nUSER: {c} Does this description match the background knowledge of \"{background}\"? Let's think step by step.\nASSISTANT:'''
                prompts.append(prompt)
    # 少样本设定提示词（2个样本）
    else:
        if withCoT == 'n':  # 不使用思维链
            prompts = []
            for i, c in zip(images, captions):
                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                answer_1 = 'Yes' if sample_1.split('/')[-1][2] == 't' else 'No'  # 样本答案
                answer_2 = 'Yes' if sample_2.split('/')[-1][2] == 't' else 'No'
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough\".\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{answer_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{answer_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
        else:  # 使用思维链
            prompts = []
            for i, c in zip(images, captions):

                background_0 = i.split('/')[-2]
                answer_0 = 'Yes' if i.split('/')[-1][2] == 't' else 'No'
                # 取两个语境样本
                filtered_list = [(index, item) for index, item in enumerate(images) if item != i]
                (_, sample_1), (_, sample_2) = random.sample(filtered_list, 2)  # 样本图像
                index_1, index_2 = images.index(sample_1), images.index(sample_2)  # 样本索引
                background_1, background_2 = sample_1.split('/')[-2], sample_2.split('/')[-2]  # 样本知识背景
                caption_1, caption_2 = captions[index_1], captions[index_2]  # 样本描述
                explanation_1, explanation_2 = explanations[index_1], explanations[index_2]  # 样本解释
                prompt = f'''You will be presented with a question.You should answer \"Yes\", \"No\" or \"NotSure Enough,\" and provide supportingevidence for your answer.\nUSER: Here are some examples:\nQ:{caption_1} Does this description match the background knowledge of \"{background_1}\"?\nA:{explanation_1}\nQ:{caption_2} Does this description match the background knowledge of \"{background_2}\"?\nA:{explanation_2}\nQ:{c} Does this description match the background knowledge of \"{background_0}\"?\nASSISTANT:'''
                prompts.append(prompt)
    # 模型推理
    result = {}
    for image, prompt in zip(images, prompts):
        image_id = image.split('/')[-1].split('.')[0]
        if "f" in image_id:

            response = client.chat.completions.create(model='gpt-3.5-turbo', messages=[{"role": "user", "content": prompt}],temperature=0.5, top_p=0)
            output = response.choices[0].message.content
            result[image_id] = output
            print(image_id, output)


    # 结果保存
    with open("../results/pipeline_explanation/"+model_path.split('/')[-1]+'-'+setting+'-'+withCoT+'-'+ispre+'.json', 'w') as file:
        json.dump(result, file, indent=4)


# 模型文件路径
PATH = {
    'llama-2-7b': '../models/Llama-2-7b',
    'llama-2-13b': '../models/Llama-2-13b',
    'vicuna-1.5-7b': '../models/vicuna-7b-v1.5',
    'vicuna-1.5-13b': '../models/vicuna-13b-v1.5',
    'GPT-3.5': 'sk-XXXXXXXXXXXXXXXXXXXXXXX'
}


# 模型调用函数
FUNCTION = {
    'llama-2-7b': llama,
    'llama-2-13b': llama,
    'vicuna-1.5-7b': vicuna,
    'vicuna-1.5-13b': vicuna,
    'GPT-3.5': gpt
}



if __name__ == '__main__':
    # 获取控制台参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", 
        type=str, 
        required=True, 
        choices=list(PATH.keys())
    )
    parser.add_argument(
        "-setting", 
        type=str, 
        required=True, 
        choices=['z','f']
    )
    parser.add_argument(
        "-withCoT", 
        type=str, 
        required=True, 
        choices=['y', 'n']
    )
    parser.add_argument(
        "-ispre", 
        type=str, 
        required=True, 
        choices=['yes', 'no']
    )
    args = parser.parse_args()
    # 模型路径
    model = PATH[args.model]
    # 图像数据集和提示词
    images, captions, explanations = load_dataset("../dataset", args.ispre)
    # print(captions)
    # 模型推理
    FUNCTION[args.model](
        model_path=model, 
        images=images, 
        captions=captions,
        explanations=explanations,
        setting=args.setting,
        withCoT=args.withCoT,
        ispre=args.ispre
    )
