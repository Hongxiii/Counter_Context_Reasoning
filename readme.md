# 一、目录结构
```
├── models：模型文件
├── datasets：数据集文件
│   ├── images：图像
│   └── annotation.xlsx：标注
├── database：知识库文件
├── results：推理结果文件
│   ├── image_caption：图像描述结果
│   ├── question_answer：视觉问答结果
│   ├── image_identification：图像识别结果
│   └── image_explanation：图像解释结果
├── baseline：基线程序
│   ├── LLaVA：LLaVA官方代码
│   ├── mPLUG-Owl：mPLUG-Owl官方代码
│   ├── mPLUG-Owl2：mPLUG-Owl2官方代码
│   ├── Otter：Otter官方代码
│   ├── openflamingo：openflamingo官方代码
│   ├── MIC：MMICL官方代码
│   ├── llama：llama官方代码
│   ├── FastChat：vicuna官方代码
│   ├── demo：演示推理代码
│   ├── infer_image_caption.py：图像描述推理程序
│   ├── infer_question_answer.py：视觉问答推理程序
│   ├── infer_image_identification.py：图像识别推理程序
│   ├── infer_image_explanation.py：图像解释推理程序
│   └── pipeline.py：流水线方法推理程序
├── ours：方法程序
│   ├── database_construct.py：知识库构建程序
│   ├── retrieval_augment_generation.py：检索增强生成程序
│   └── object_detection.py：目标检测程序
├── tools：工具程序
│   ├── generate_vqa.py：vqa数据生成程序
│   ├── preprocess.py：数据预处理程序
│   └── download.py：模型下载程序
└── evaluate：评估程序
│   ├── eval_image_caption.py：图像描述评估程序
│   ├── eval_question_answer.py：视觉问答评程序
│   └── eval_image_explanation.py：图像解释评估程序
└── result：预测结果
│   ├── image_caption：图像描述结果，对应论文表1
│   ├── question_answer：视觉问答结果，对应论文表1
│   ├── pipeline_identification：图像识别结果（流水线），对应论文表2
│   ├── pipeline_explanation：图像解释结果（流水线），对应论文表2
│   ├── image_identification：图像识别结果（端到端），对应论文表3
│   └── image_explanation：图像解释结果（端到端），对应论文表3
└── readme.md：说明文件
```

# 二、运行

## 1. 模型下载
```shell
export HF_ENDPOINT=https://hf-mirror.com
cd main
python download.py
```

## 2. 数据预处理
```shell
python preprocess.py -task caption
python preprocess.py -task explanation
python preprocess.py -task vqa
```

## 3. 环境部署
BLIP系列、BLIP2系列、InstrcutBLIP系列环境：
```shell
conda activate blip
```
mPLUG-Owl系列环境：
```shell
conda activate mplug_owl
```
mPLUG-Owl2系列环境：
```shell
conda activate mplug_owl2
```
LLaVA系列、openflamingo系列环境：
```shell
conda activate llava
```
LLaMA系列环境：
```shell
conda activate llama
```
Vicuna系列环境：
```shell
conda activate vicuna
```
ours环境：
```shell
conda activate cfr
```

## 4. baseline推理
【注意】：4个任务的实验测试输入略有不同。
- <u>图像描述</u>和<u>视觉问答</u>是针对所有图像测试的，包括正样本和负样本。只有zero-shot设置。
- <u>图像识别</u>是针对所有图像进行测试的。在few-shot设置下，除了读取测试图像，还需要读取与测试图像同属一个知识背景的2个随机样本（可能是正样本也可能是负样本）；在CoCoT设置下，除了读取测试样本，还需要读取对应的1个相反样本。
- <u>图像解释</u>是针对负样本图像进行测试的。在few-shot设置下，除了读取测试图像，还需要读取与测试图像同属一个知识背景的2个随机样本（可能是正样本也可能是负样本）；在CoCoT设置下，除了读取测试样本，还需要读取对应的1个相反样本（正样本）。
- GPT-4V是一个例外。对<u>图像解释</u>的结果进行后处理，得到<u>图像识别</u>的结果，因此它的<u>图像识别</u>也是针对负样本进行测试的。
- pipeline方法中，few-shot设置下选取的样本是从全部数据集中抽取的两个样本（可能是正样本，可能是负样本，也可能是其他的知识背景）。


1. 图像描述推理

    指定模型，执行命令：
    ```shell
    python infer_image_caption.py -model BLIP-Base
    ```
2. 视觉问答推理

    指定模型，执行命令：
    ```shell
    python infer_question_answer.py -model BLIP-Base
    ```
3. 图像识别推理

    指定模型，执行命令：
    ```shell
    python infer_image_identification.py -model BLIP2-XL -setting z
    ```
4. 图像解释推理

    指定模型，执行命令：
    ```shell
    python infer_image_explanation.py -model BLIP2-XL -setting z
    ```
5. 流水线方法推理
    指定模型，执行命令：
    ```shell
    python pipeline.py -model LLaMA-2-7B -setting z -withCoT n
    ```

图像描述的实验模型：
| 可选模型 | 模型文件 |
|:------ |:-------|
| BLIP-Base | ./models/blip-image-captioning-base |
| BLIP2-XL | ./models/blip2-flan-t5-xl |
| BLIP2-XXL | ./models/blip2-flan-t5-xxl |
| InstructBLIP-XL | ./models/instructblip-flan-t5-xl |
| InstructBLIP-XXL | ./models/instructblip-flan-t5-xl |
| mPLUG-owl-7B | ./models/mplug-owl-llama-7b |
| mPLUG-owl2-7B | ./models/mplug-owl2-llama-7b |
| LLaVA-1.5-7B | ./models/llava-v1.5-7b |
| LLaVA-1.6-7B | ./models/llava-v1.6-vicuna-7b |

视觉问答的实验模型：
| 可选模型 | 模型文件 |
|:------ |:-------|
| BLIP-Base | ./models/blip-vqa-base |
| BLIP2-XL | ./models/blip2-flan-t5-xl |
| BLIP2-XXL | ./models/blip2-flan-t5-xxl |
| InstructBLIP-XL | ./models/instructblip-flan-t5-xl |
| InstructBLIP-XXL | ./models/instructblip-flan-t5-xl |
| mPLUG-owl-7B | ./models/mplug-owl-llama-7b |
| mPLUG-owl2-7B | ./models/mplug-owl-llama2-7b |
| LLaVA-1.5-7B | ./models/llava-v1.5-7b |
| LLaVA-1.6-7B | ./models/llava-v1.6-vicuna-7b |

图像识别及图像解释的实验模型：
| 可选模型 | 模型文件/api_key | 实验设置 |
|:------ |:-------|:-------|
| BLIP2-XL | ./models/blip2-flan-t5-xl | zero-shot |
| BLIP2-XXL | ./models/blip2-flan-t5-xxl | zero-shot |
| InstructBLIP-XL | ./models/instructblip-flan-t5-xl | zero-shot |
| InstructBLIP-XXL | ./models/instructblip-flan-t5-xl | zero-shot |
| mPLUG-owl-7B | ./models/mplug-owl-llama-7b | zero-shot |
| mPLUG-owl2-7B | ./models/mplug-owl2-llama-7b | zero-shot |
| LLaVA-1.5-7B | ./models/llava-v1.5-7b | zero-shot |
| LLaVA-1.6-7B | ./models/llava-v1.6-vicuna-7b | zero-shot |
| MMICL | ./models/MMICL-Instructblip-T5-xl  | few-shot, CoCoT |
| OpenFlamingo | ./models/OpenFlamingo-3B-vitl-mpt1b | few-shot, CoCoT |
| Otter-7B | ./models/OTTER-Image-LLaMA7B-LA-InContext | few-shot, CoCoT |
| GEMINI | coming soon... | few-shot, CoCoT |
| GPT-4V | sk-icjltWAeAZCp0oMNAcD970B5F88546169515B8995e66C389 | few-shot, CoCoT |

流水线方法的实验模型：
| 可选模型 | 模型文件 |
|:------ |:-------|
| llama-2-7b | ./models/Llama-2-7b-hf |
| llama-2-13b | ./models/Llama-2-13b-hf |
| vicuna-1.5-7b  | ./models/vicuna-7b-v1.5 |
| vicuna-1.5-7b | ./models/vicuna-13b-v1.5 |
| GPT-3.5 | sk-6H72K8DB8HOw0u8u1768Cc8e1fEa438284Ee38C807A7E76a |

## 5. ours method推理


# 性能评估
1. 图像描述评估
2. 视觉问答评估
3. 图像识别评估
4. 图像解释评估