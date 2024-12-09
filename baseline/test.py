import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import sys
import pandas as pd
from collections import defaultdict
import torch
import re

import base64

import random
from accelerate import Accelerator



def llava(model_path, dataset_path, prompts):
    from LLaVA.llava.model.builder import load_pretrained_model
    from LLaVA.llava.mm_utils import get_model_name_from_path
    from LLaVA.llava.eval.run_llava import eval_model
    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    # 推理图像
    result = {}
    image = '../dataset/images/Fairytale/Little Red Riding Hood/b-t-1-5.webp'
    prompt = 'Does this image match the story of “Little Red Riding Hood”?'
    args = type(
        'Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 128
    })()
    output = eval_model(args)
    print(output)



# 加载图像和提示词
def load_dataset(dataset_path):
    # 读取图像路径
    images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # 只读取错误图像
            if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="f":
                images.append(os.path.join(root, file))
    # 读取图像故事
    prompts = []
    annotation_file = dataset_path + "/dataset.xlsx"
    annotation_data = pd.ExcelFile(annotation_file)
    data = annotation_data.parse(annotation_data.sheet_names[2])
    data = data.iloc[:, 0:2].set_index(data.columns[0])[data.columns[1]].to_dict()
    # 统一列表顺序
    for i in images:
        image_id = os.path.splitext(os.path.basename(i))[0]
        prompt = 'Explain why this image does not correspond to the \"' + data[image_id] + '\"?'
        prompts.append(prompt)
    return images, prompts



 
if __name__ == '__main__':
    # 模型路径
    model = '../models/llava-v1.5-7b'
    # 图像数据集和提示词
    images, prompts = load_dataset("../dataset")
    # 模型推理
    llava(model, images, prompts)
