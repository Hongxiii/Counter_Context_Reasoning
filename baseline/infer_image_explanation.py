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


def blip2(model_path, dataset_path, prompts, setting):
    if setting == 'z':
        print("Unsupported experiment setting!!!")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    # 加载模型
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    # 推理图像
    result = {}
    for image, prompt in tqdm(zip(dataset_path, prompts)):
        print(prompt)
        raw_image = Image.open(image).convert('RGB')
        inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        image_id = os.path.splitext(image)[0].split('/')[-1]
        result[image_id] = processor.decode(out[0], skip_special_tokens=True)
    # 保存结果
    with open('../results/image_explanation/'+model_path.split('/')[-1]+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def instructblip(model_path, dataset_path, prompts, setting):
    if setting == 'z':
        print("Unsupported experiment setting!!!")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    # 加载模型
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path).to("cuda")
    processor = InstructBlipProcessor.from_pretrained(model_path)
    # 推理图像
    result = {}
    for image, prompt in tqdm(zip(dataset_path, prompts)):
        raw_image = Image.open(image).convert('RGB')
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        image_id = os.path.splitext(image)[0].split('/')[-1]
        result[image_id] = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    # 保存结果
    with open('../results/image_explanation/'+model_path.split('/')[-1]+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def mplug_owl(model_path, dataset_path, prompts, setting):
    if setting == 'z':
        print("Unsupported experiment setting!!!")
        sys.exit()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    from mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mPLUG_Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
    from transformers import AutoTokenizer
    # 加载模型
    pretrained_ckpt = model_path
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    # 推理图像
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
    }
    result = {}
    model.to('cuda')
    for image, prompt in tqdm(zip(dataset_path, prompts)):
        prompt = [
            "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite explanations to the user's questions."
            +"\nHuman: <image>"
            +"\nHuman: "
            +prompt 
            +"\nAI: "
        ]
        image = [image]
        images = [Image.open(_) for _ in image]
        inputs = processor(text=prompt, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)
        image_id = os.path.splitext(image[0])[0].split('/')[-1]
        result[image_id] = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    # 保存结果
    with open('../results/image_explanation/'+model_path.split('/')[-1]+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def mplug_owl2(model_path, dataset_path, prompts, setting):
    if setting == 'z':
        print("Unsupported experiment setting!!!")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from transformers import TextStreamer
    from mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from mPLUG_Owl2.mplug_owl2.conversation import conv_templates, SeparatorStyle
    from mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
    from mPLUG_Owl2.mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    def call(image_file, prompt):
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        image = Image.open(image_file).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
    
    # 推理图像
    result = {}
    for image, prompt in tqdm(zip(dataset_path, prompts)):
        output = call(image, prompt)
        image_id = os.path.splitext(image)[0].split('/')[-1]
        result[image_id] = output
    # 保存结果
    with open('../results/image_explanation/'+model_path.split('/')[-1]+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def llava(model_path, dataset_path, prompts, setting):
    if setting == 'z':
        print("Unsupported experiment setting!!!")
        sys.exit()
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
    for image, prompt in tqdm(zip(dataset_path, prompts)):
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
        image_id = os.path.splitext(image)[0].split('/')[-1]
        result[image_id] = output
    # 保存结果
    with open('../results/image_explanation/'+model_path.split('/')[-1]+'.json', 'w') as file:
        json.dump(result, file, indent=4)


def openflamingo(model_path, dataset_path, prompts, setting):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from Flamingo.open_flamingo import create_model_and_transforms
    # 加载模型
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="../models/mpt-1b-redpajama-200b",
        tokenizer_path="../models/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
    )
    model.load_state_dict(torch.load(model_path + "/checkpoint.pt", map_location="cuda"), strict=False)
    model.to("cuda")
    tokenizer.padding_side = "left"
    if setting == 'f':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        dataset_path.extend(samples)  # 正负样本合并
        explanations = []
        annotation_file = "../dataset/dataset.xlsx"
        annotation_data = pd.ExcelFile(annotation_file)
        explanation_data = annotation_data.parse(annotation_data.sheet_names[2])
        explanation_data = explanation_data.iloc[:, 0:7].set_index(explanation_data.columns[0])[explanation_data.columns[6]].to_dict()
        for i in dataset_path:
            image_id = os.path.splitext(os.path.basename(i))[0]
            if image_id in explanation_data:
                explanation = explanation_data[image_id]
                explanation = re.findall(r'\[(.*?)\]', explanation)[0]  # 读取了第一条标注解释
            else:
                explanation = "This image aligns with \'" + i.split('/')[-2] + "\'."
            explanations.append(explanation)

        images = defaultdict(list)
        answers = defaultdict(list)
        for image, answer in zip(dataset_path, explanations):
            category = image.split('/')[-2]
            images[category].append(image)
            answers[category].append(answer)
        images = list(images.values())
        answers = list(answers.values())
        # 推理图像
        result = {}
        for image, answer in tqdm(zip(images, answers)):
            for i in image:
                if i.split('/')[-1][2] == 'f':  # 仅对负样本进行推理
                    # 随机抽取两个上下文样本
                    image_samples = [(index, value) for index, value in enumerate(image) if value != i]
                    samples = random.sample(image_samples, 2)
                    sample_index_one, sample_image_one = samples[0]
                    sample_index_two, sample_image_two = samples[1]
                    sample_prompt_one, sample_prompt_two = answer[sample_index_one],  answer[sample_index_two]
                    # 读取图像和提示词
                    # print(i)
                    # print(sample_image_one, sample_prompt_one)
                    # print(sample_image_two,sample_prompt_two)
                    # print("*"*100)
                    sample_image_one = Image.open(sample_image_one)
                    sample_image_two = Image.open(sample_image_two)
                    query_image = Image.open(i)

                    lang_x = tokenizer(
                        ["<image>" + sample_prompt_one + "<|endofchunk|><image>" + sample_prompt_two + "<|endofchunk|><image>This image"],
                        return_tensors="pt",
                    ).to("cuda")
                    vision_x = [image_processor(sample_image_one).unsqueeze(0), image_processor(sample_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
                    # 语境学习
                    vision_x = torch.cat(vision_x, dim=0)
                    vision_x = vision_x.unsqueeze(1).unsqueeze(0).to("cuda")
                    generated_text = model.generate(
                        vision_x=vision_x,
                        lang_x=lang_x["input_ids"],
                        attention_mask=lang_x["attention_mask"],
                        max_new_tokens=128,
                        num_beams=3,
                    )
                    output = tokenizer.decode(generated_text[0])

                    output = output.split("<|endofchunk|><image>")
                    output = output[-1].replace("<|endofchunk|>", "")
                    image_id = os.path.splitext(i)[0].split('/')[-1]
                    result[image_id] = output
                    print(image_id, output)
        # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    elif setting == 'c':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 随机抽取一个正样本
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理图像
        result = {}
        for image, incontext in tqdm(zip(dataset_path, incontexts)):
            incontext_prompt = incontext.split('/')[-2]
            incontext_prompt = 'This image is consistent with the \"' + incontext_prompt + '\" story, as every detail in the picture aligns with the relevant background knowledge.'
            # 读取图像和提示词

            incontext = Image.open(incontext)
            query_image = Image.open(image)
            lang_x = tokenizer(
                ["<image>" + incontext_prompt + "<|endofchunk|><image>This image"],
                return_tensors="pt",
            ).to("cuda")
            vision_x = [image_processor(incontext).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
            # 语境学习
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).to("cuda")
            generated_text = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=128,
                num_beams=3,
            )
            output = tokenizer.decode(generated_text[0])
            output = output.split("<|endofchunk|><image>")

            output = output[-1].replace("<|endofchunk|>","")
            image_id = os.path.splitext(image)[0].split('/')[-1]
            result[image_id] = output
            print(image_id, output)
            # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    elif setting == 'z':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 随机抽取一个正样本
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理图像
        result = {}
        for image, incontext in tqdm(zip(dataset_path, incontexts)):

            query_image = Image.open(image)
            background = image.split('/')[-2]
            lang_x = tokenizer(
                ['<image> Question: Explain why this image does not align with the background knowledge of \"' + background + '\". Just answer the reason. Answer:'],
                return_tensors="pt",
            ).to("cuda")
            vision_x = [image_processor(query_image).unsqueeze(0)]
            # 语境学习
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).to("cuda")
            generated_text = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=128,
                num_beams=3,
            )
            output = tokenizer.decode(generated_text[0])
            image_id = os.path.splitext(image)[0].split('/')[-1]
            # output = output.replace(prompt, "").replace("<|endofchunk|>","")
            output = output.split("Answer:")[-1]
            result[image_id] = output
            print(image_id, output)
            # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)

    else:
        print("Unsupported experiment setting!!!")
        sys.exit()


def mmicl(model_path, dataset_path, prompts, setting):
    from MIC.model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
    import transformers
    # 加载模型
    processor_ckpt = "../models/instructblip-flan-t5-xl"
    config = InstructBlipConfig.from_pretrained(model_path)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_path,
        config=config).to('cuda:0',dtype=torch.bfloat16) 
    image_palceholder="图"
    sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
    processor = InstructBlipProcessor.from_pretrained(
        processor_ckpt
    )
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    replace_token="".join(32*[image_palceholder])
    if setting == 'f':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        dataset_path.extend(samples)  # 正负样本合并
        explanations = []
        annotation_file = "../dataset/dataset.xlsx"
        annotation_data = pd.ExcelFile(annotation_file)
        explanation_data = annotation_data.parse(annotation_data.sheet_names[2])
        explanation_data = explanation_data.iloc[:, 0:7].set_index(explanation_data.columns[0])[explanation_data.columns[6]].to_dict()
        for i in dataset_path:
            image_id = os.path.splitext(os.path.basename(i))[0]
            if image_id in explanation_data:
                explanation = explanation_data[image_id]
                explanation = re.findall(r'\[(.*?)\]', explanation)[0]  # 读取了第一条标注解释
            else:
                explanation = "This image aligns with \'" + i.split('/')[-2] + "\'."
            explanations.append(explanation)
        images = defaultdict(list)
        answers = defaultdict(list)
        for image, answer in zip(dataset_path, explanations):
            category = image.split('/')[-2]
            images[category].append(image)
            answers[category].append(answer)
        images = list(images.values())
        answers = list(answers.values())
        # 推理图像
        result = {}
        for image, answer in tqdm(zip(images, answers)):
            for i in image:
                if i.split('/')[-1][2] == 'f':  # 仅对负样本进行推理
                    # 随机抽取两个上下文样本
                    image_samples = [(index, value) for index, value in enumerate(image) if value != i]
                    samples = random.sample(image_samples, 2)
                    sample_index_one, sample_image_one = samples[0]
                    sample_index_two, sample_image_two = samples[1]
                    sample_prompt_one, sample_prompt_two = answer[sample_index_one],  answer[sample_index_two]
                    # 读取图像和提示词
                    sample_image_one = Image.open(sample_image_one)
                    sample_image_two = Image.open(sample_image_two)
                    query_image = Image.open(i)
                    background = i.split('/')[-2]
                    images = [sample_image_one,sample_image_two,query_image]
                    prompt = [f'image 0 is <sample_image_one>{replace_token},image 1 is <sample_image_two>{replace_token},image 2 is <query_image>{replace_token}.Question: <sample_image_one> Does this image align with {background}. If not, why? Answer: {sample_prompt_one}\n Question: <sample_image_two> Does this image align with {background}?. If not, why? Answer: {sample_prompt_two}\n Question: <query_image> Does this image align with {background}? If not, why? Answer:'
                    ]
                    prompt = " ".join(prompt)
                    # print(prompt)
                    inputs = processor(images=images, text=prompt, return_tensors="pt")
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                    inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
                    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                    inputs = inputs.to('cuda:0')
                    outputs = model.generate(
                            pixel_values = inputs['pixel_values'],
                            input_ids = inputs['input_ids'],
                            attention_mask = inputs['attention_mask'],
                            img_mask = inputs['img_mask'],
                            do_sample=False,
                            max_length=512,
                            min_length=1,
                            set_min_padding_size =False,
                    )
                    output = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    
                    # sys.exit(0)
                    image_id = os.path.splitext(i)[0].split('/')[-1]
                    result[image_id] = output
                    print(image_id,output)
        # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    elif setting == 'c':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 随机抽取一个正样本
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理图像
        result = {}
        for image, incontext in tqdm(zip(dataset_path, incontexts)):
            incontext_prompt = incontext.split('/')[-2]
            incontext_prompt = 'This image is consistent with the \"' + incontext_prompt + '\" story, as every detail in the picture aligns with the relevant background knowledge.'
            # 读取图像和提示词
            incontext = Image.open(incontext)
            query_image = Image.open(image)
            images = [incontext,query_image]
            background = image.split('/')[-2]
            prompt = [f'Use the image 0: <incontext>{replace_token}and image 1: <query_image>{replace_token}. Question: Review two images, image 0 correctly depicts \"' + background + '\". Examine both the similarities and differences between these images to assess whether image 1 aligns with the background knowledge of \"' + background + '\". If it does not, explain why. Answer:']
            prompt = " ".join(prompt)

            inputs = processor(images=images, text=prompt, return_tensors="pt")

            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

            inputs = inputs.to('cuda:0')
            outputs = model.generate(
                    pixel_values = inputs['pixel_values'],
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    img_mask = inputs['img_mask'],
                    do_sample=False,
                    max_length=200,
                    min_length=50,
                    num_beams=8,
                    set_min_padding_size =False,
            )
            output = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            image_id = os.path.splitext(image)[0].split('/')[-1]
            result[image_id] = output
            print(image_id, output)
            # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    elif setting == 'z':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 随机抽取一个正样本
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理图像
        result = {}
        for image, incontext in tqdm(zip(dataset_path, incontexts)):

            query_image = Image.open(image)
            images = [query_image]
            background = image.split('/')[-2]
            prompt = [f'Use the image : <query_image>{replace_token}. Question: Explain why this image does not align with the background knowledge of \"' + background + '\". Just answer the reason. Answer:']
            prompt = " ".join(prompt)

            inputs = processor(images=images, text=prompt, return_tensors="pt")

            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

            inputs = inputs.to('cuda:0')
            outputs = model.generate(
                    pixel_values = inputs['pixel_values'],
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    img_mask = inputs['img_mask'],
                    do_sample=False,
                    max_length=200,
                    min_length=50,
                    num_beams=8,
                    set_min_padding_size =False,
            )
            output = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            image_id = os.path.splitext(image)[0].split('/')[-1]
            result[image_id] = output
            print(image_id, output)
            # 保存结果
        with open('../results/image_explanation/'+model_path.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    else:
        print("Unsupported experiment setting!!!")
        sys.exit()


def gpt_4v(api_key, dataset_path, prompts, setting):
    from openai import OpenAI
    import httpx

    def encode_image_to_base64(image_path): 
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    client = OpenAI(
        base_url="https://oneapi.xty.app/v1", 
        api_key=api_key,
        http_client=httpx.Client(
            base_url="https://oneapi.xty.app/v1",
            follow_redirects=True,
        ),
    )

    if setting == 'f':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        dataset_path.extend(samples)  # 正负样本合并
        explanations = []
        annotation_file = "../dataset/dataset.xlsx"
        annotation_data = pd.ExcelFile(annotation_file)
        explanation_data = annotation_data.parse(annotation_data.sheet_names[2])
        explanation_data = explanation_data.iloc[:, 0:7].set_index(explanation_data.columns[0])[explanation_data.columns[6]].to_dict()
        for i in dataset_path:
            image_id = os.path.splitext(os.path.basename(i))[0]
            if image_id in explanation_data:
                explanation = explanation_data[image_id]
                explanation = re.findall(r'\[(.*?)\]', explanation)[0]  # 读取了第一条标注解释
            else:
                explanation = "This image aligns with \'" + i.split('/')[-2] + "\'."
            explanations.append(explanation)

        images = defaultdict(list)
        answers = defaultdict(list)
        for image, answer in zip(dataset_path, explanations):
            category = image.split('/')[-2]
            images[category].append(image)
            answers[category].append(answer)
        images = list(images.values())
        answers = list(answers.values())
        # 推理
        result = {}
        
        with open("/data/lihongxi/Counter-Factual-Reasoning/results/image_explanation/sk-hBAtTjpW4F6vEcyTB8De1577Af1d4cFf9d061502C56f390c-f.json", 'r') as f:
            data22 = json.load(f)
        for image, answer in tqdm(zip(images, answers)):
            for i in image:
                if i.split('/')[-1][2] == 'f':
                    # 随机抽取两个上下文样本
                    image_samples = [(index, value) for index, value in enumerate(image) if value != i]
                    samples = random.sample(image_samples, 2)
                    sample_index_one, sample_image_one = samples[0]
                    sample_index_two, sample_image_two = samples[1]
                    sample_prompt_one, sample_prompt_two = answer[sample_index_one],  answer[sample_index_two]
                    # 输入图像格式转化
                    # print(sample_prompt_one)
                    image_one = encode_image_to_base64(sample_image_one)
                    image_two = encode_image_to_base64(sample_image_two)
                    query_image = encode_image_to_base64(i)
                    image_one_format = image_one.split('/')[-1].split('.')[-1]
                    image_two_format = image_two.split('/')[-1].split('.')[-1]
                    query_image_format = query_image.split('/')[-1].split('.')[-1]
                    background = i.split('/')[-2]
                    image_id = os.path.splitext(i)[0].split('/')[-1]
                    if image_id in data22.keys() and data22[image_id][0]!="I":
                        continue
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4-vision-preview",
                            messages=[
                                {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": 'Based on the following sample that includes images and their corresponding questions and answers, predict the explanation of the third image.'
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{image_one_format};base64,{image_one}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f'Q: Does this image conform to the knowledge background of \"' + background + f'\"? If no, please provide the reasons. A: {sample_prompt_one}'
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{image_two_format};base64,{image_two}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f'Q: Does this image conform to the knowledge background of \"' + background + f'\"? If no, please provide the reasons. A: {sample_prompt_two}'
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{query_image_format};base64,{query_image}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f'Q: This image does not conform to the knowledge background of \"' + background + f'\". Please provide the reasons. Try your best! A: '
                                    },
                                ],
                                }
                            ],
                            max_tokens=300,
                        )
                        output = response.choices[0].message.content
                        image_id = os.path.splitext(i)[0].split('/')[-1]
                        result[image_id] = output
                        print(image_id, output)
                    except:
                        
                        print("pass")

        with open('../results/image_explanation/'+api_key.split('/')[-1]+'-'+setting+'.json', 'a') as file:
            json.dump(result, file, indent=4)
    elif setting == 'c':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 从同属一个类别的正样本中随机抽取
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理
        result = {}
        for image, incontext in tqdm(zip(dataset_path, incontexts)):
            image_one = encode_image_to_base64(incontext)
            image_two = encode_image_to_base64(image)
            background = image.split('/')[-2]
            image_one_format = incontext.split('/')[-1].split('.')[-1]
            image_two_format = image.split('/')[-1].split('.')[-1]
            try: 
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": 'Review two images: the first correctly depicts \"' + background + '\". Examine both the similarities and differences between these images to assess why the second image does not align with the background knowledge of \"' + background + '\". Just answer the reason why the second image does not align with the background knowledge of \"' + background + '\".'
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_one_format};base64,{image_one}",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_two_format};base64,{image_two}",
                                },
                            },
                        ],
                        }
                    ],
                    max_tokens=300,
                )
                output = response.choices[0].message.content
                image_id = os.path.splitext(image)[0].split('/')[-1]
                result[image_id] = output
            except:
                break
        with open('../results/image_explanation/'+api_key.split('/')[-1]+'-'+setting+'.json', 'w') as file:
            json.dump(result, file, indent=4)
    elif setting=='z':
        # 重载数据
        samples = []
        for root, dirs, files in os.walk('../dataset'):
            for file in files:
                # 只读取正确图像
                if file.endswith(('.png', '.webp', '.jpg')) and file.split("-")[1]=="t":
                    samples.append(os.path.join(root, file))
        incontexts = []
        for image in dataset_path:
            story = image.split('/')[4]
            # 从同属一个类别的正样本中随机抽取
            incontext = random.choice([s for s in samples if s.split('/')[4] == story])
            incontexts.append(incontext)
        # 推理
        result = {}
        with open('/data/lihongxi/Counter-Factual-Reasoning/results/image_explanation/sk-hBAtTjpW4F6vEcyTB8De1577Af1d4cFf9d061502C56f390c-z.json' ,'r') as file:
            exit_data = json.load(file)
        
        for image, incontext in tqdm(zip(dataset_path, incontexts)):
            image_two = encode_image_to_base64(image)
            background = image.split('/')[-2]
            image_id = os.path.splitext(image)[0].split('/')[-1]
            if image_id in exit_data.keys():
                continue
            image_two_format = image.split('/')[-1].split('.')[-1]
            try: 
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": 'Explain why this image does not align with the background knowledge of \"' + background + '\". Just answer the reason.'
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_two_format};base64,{image_two}",
                                },
                            },
                        ],
                        }
                    ],
                    max_tokens=300,
                )
                output = response.choices[0].message.content
                image_id = os.path.splitext(image)[0].split('/')[-1]
                result[image_id] = output
                print(image_id, output)
            except:
                break
        with open('../results/image_explanation/'+api_key.split('/')[-1]+'-'+setting+'.json', 'a+') as file:
            json.dump(result, file, indent=4)
    else:
        print("Unsupported experiment setting!!!")
        sys.exit()
    

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


# 模型文件路径
PATH = {
    'BLIP2-XL': '../models/blip2-flan-t5-xl',
    'BLIP2-XXL': '../models/blip2-flan-t5-xxl',
    'InstructBLIP-XL': '../models/instructblip-flan-t5-xl',
    'InstructBLIP-XXL': '../models/instructblip-flan-t5-xxl',
    'mPLUG-owl-7B': '../models/mplug-owl-llama-7b',
    'mPLUG-owl2-7B': '../models/mplug-owl2-llama2-7b',
    'LLaVA-1.5-7B': '../models/llava-v1.5-7b',
    'LLaVA-1.6-7B': '../models/llava-v1.6-vicuna-7b',
    'OpenFlamingo': '../models/OpenFlamingo-3B-vitl-mpt1b',
    'MMICL': '../models/MMICL-Instructblip-T5-xl',
    'GPT-4V': 'sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
}

# 模型调用函数
FUNCTION = {
    'BLIP2-XL': blip2,
    'BLIP2-XXL': blip2,
    'InstructBLIP-XL': instructblip,
    'InstructBLIP-XXL': instructblip,
    'mPLUG-owl-7B': mplug_owl,
    'mPLUG-owl2-7B': mplug_owl2,
    'LLaVA-1.5-7B': llava,
    'LLaVA-1.6-7B': llava,
    'OpenFlamingo': openflamingo,
    'MMICL': mmicl,
    'GPT-4V': gpt_4v
}

 
if __name__ == '__main__':
    # 获取控制台参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, required=True, choices=list(PATH.keys())
    )
    parser.add_argument(
        "-setting", type=str, required=True, choices=['z','f','c'],
        help="z=zero-shot, f=few-shot, c=CoCoT"
    )
    args = parser.parse_args()
    # 模型路径
    model = PATH[args.model]
    # 图像数据集和提示词
    images, prompts = load_dataset("../dataset")
    # 模型推理
    FUNCTION[args.model](model, images, prompts, args.setting)
