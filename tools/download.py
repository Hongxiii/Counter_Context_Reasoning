from huggingface_hub import snapshot_download
from tqdm import tqdm


model_list = [
    # "ToughStone/BLEURT-20",
    # "Salesforce/blip-image-captioning-base",
    # "Salesforce/blip-vqa-base",
    # "Salesforce/blip2-flan-t5-xl",
    # "Salesforce/blip2-flan-t5-xxl",
    # "Salesforce/instructblip-flan-t5-xl",
    # "Salesforce/instructblip-flan-t5-xxl",
    # "MAGAer13/mplug-owl-llama-7b",
    # "MAGAer13/mplug-owl2-llama2-7b",
    # "liuhaotian/llava-v1.5-7b",
    # "liuhaotian/llava-v1.6-vicuna-7b",
    # "openai/clip-vit-large-patch14-336",
    # "luodian/OTTER-Image-LLaMA7B-LA-InContext",
    # "luodian/llama-7b-hf",
    # "openai/clip-vit-large-patch14",
    # "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    # "anas-awadalla/mpt-1b-redpajama-200b",
    # "BleachNick/MMICL-Instructblip-T5-xl",
    # "meta-llama/Llama-2-7b",
    # "meta-llama/Llama-2-13b",
    # "lmsys/vicuna-7b-v1.5",
    # "lmsys/vicuna-13b-v1.5",
    "GLIPModel/GLIP"
]


def download(model_name):
    finish = False
    while not finish:
        if model_name == "liuhaotian/llava-v1.6-vicuna-7b":
            snapshot_download(
                repo_id=model_name,
                local_dir='../models/' + model_name.split('/')[-1],
                ignore_patterns=["*.h5"]
            )
        elif model_name == "GLIPModel/GLIP":
            snapshot_download(
                repo_id=model_name,
                local_dir='../models/' + model_name.split('/')[-1],
                allow_patterns=["*.pth"]
            )
        elif model_name.split('/')[0] == "meta-llama":
            snapshot_download(
                token='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',
                repo_id=model_name,
                local_dir='../models/' + model_name.split('/')[-1],
                ignore_patterns=["model.safetensors.index.json","*.safetensors", "*.h5"]
            )
        else:
            snapshot_download(
                repo_id=model_name,
                local_dir='../models/' + model_name.split('/')[-1],
                ignore_patterns=["model.safetensors.index.json","*.safetensors", "*.h5"]
            )
        finish = True


if __name__ == '__main__':
    for model in tqdm(model_list):
        print('==================================================')
        print('Model: \"{}\" is currently being downloaded ...... '.format(model.split('/')[-1]))
        print('==================================================\n\n\n\n\n') 
        download(model)
        print('==================================================')
        print('Model: \"{}\" successfully downloaded to local directory \"{}\"'.format(model, '../models/' + model.split('/')[-1]))
        print('==================================================\n\n\n\n\n')
