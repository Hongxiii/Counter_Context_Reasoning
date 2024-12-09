from openai import OpenAI
import time
import os
import argparse
import json
import pandas as pd
import httpx
from tqdm import tqdm

def _ms_since_epoch():
    return time.perf_counter_ns() // 1000000


def set_openai_parameters(engine, max_tokens):

    parameters = {
        "max_tokens": max_tokens,
        "top_p": 0,  # greedy
        "temperature": 0.5,
        "logprobs": 5,  # maximal value accorrding to https://beta.openai.com/docs/api-reference/completions/create#completions/create-logprobs, used to be 10...
        "engine": engine,
    }
    time_of_last_api_call = _ms_since_epoch()

    return parameters, time_of_last_api_call


def wait_between_predictions(time_of_last_api_call, min_ms_between_api_calls):
    if (cur_time := _ms_since_epoch()) <= time_of_last_api_call + min_ms_between_api_calls:
        ms_to_sleep = min_ms_between_api_calls - (cur_time - time_of_last_api_call)
        time.sleep(ms_to_sleep / 1000)
    time_of_last_api_call = _ms_since_epoch()


def predict_sample_openai_gpt(
    example,
    prompt,
    min_ms_between_api_calls: int = 500,
    engine: str = "text-davinci-003",
    max_tokens: int = 100,
):
    parameters, time_of_last_api_call = set_openai_parameters(engine, max_tokens)
    parameters["prompt"] = prompt

    # OpenAI limits us to 3000 calls per minute:
    # https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
    # that is why the default value of min_ms_between_api_calls is 20
    wait_between_predictions(time_of_last_api_call, min_ms_between_api_calls)

    response = openai.Completion.create(**parameters)

    if response is None:
        raise Exception("Response from OpenAI API is None.")

    # build output data
    prediction = dict()
    prediction["input"] = prompt
    prediction["prediction"] = response.choices[0].text.strip().strip(".")  # type:ignore

    # build output metadata
    metadata = example.copy()  # dict()
    metadata["logprobs"] = response.choices[0]["logprobs"]  # type:ignore
    # "finish_reason" is located in a slightly different location in opt
    if "opt" in engine:
        finish_reason = response.choices[0]["logprobs"][  # type:ignore
            "finish_reason"
        ]
    else:
        finish_reason = response.choices[0]["finish_reason"]  # type:ignore
    metadata["finish_reason"] = finish_reason
    if "opt" not in engine:
        # From the OpenAI API documentation it's not clear what "index" is, but let's keep it as well
        metadata["index"] = response.choices[0]["index"]  # type:ignore

    prediction["metadata"] = metadata

    return prediction

def predict_sample_openai_chatgpt(
    prompt,
    min_ms_between_api_calls: int = 1000,
    engine: str = "gpt-4-0613",
    max_tokens: int = 100,
):
    parameters, time_of_last_api_call = set_openai_parameters(engine, max_tokens)
    parameters["prompt"] = prompt
    api_key = "sk-hkR7ohdkEqFjD9MiEc22B0C392484eE692Ca0aD3F8B06c5c"
    client = OpenAI(
        base_url="https://oneapi.xty.app/v1", 
        api_key=api_key,
        http_client=httpx.Client(
            base_url="https://oneapi.xty.app/v1",
            follow_redirects=True,
        ),
    )
    # OpenAI limits us to 3000 calls per minute:
    # https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
    wait_time = 0.2
    time.sleep(wait_time)
    try:
        response = client.chat.completions.create(model=engine, messages=[{"role": "user", "content": prompt}],
                                                temperature=parameters['temperature'], top_p=parameters['top_p'])
    except openai.error.RateLimitError as e:
        wait_time = 10
        print(f"Rate limit reached. Waiting {wait_time} seconds.")
        time.sleep(wait_time)

        response = client.chat.completions.create(model=engine, messages=[{"role": "user", "content": prompt}],
                                                temperature=parameters['temperature'], top_p=parameters['top_p'])

    if response is None:
        raise Exception("Response from OpenAI API is None.")

    # build output data
    prediction = dict()
    prediction["input"] = prompt
    prediction["prediction"] = response.choices[0].message.content  # type:ignore

    return prediction

def gpt4_estimetion(A,B,story):
  prompt= f"""
  Evaluate the equivalence of the following explanations for the question "Explain why this image does not correspond to the \'{story} \'?" Just answer with True or False:
  A: '{A}'
  B: '{B}'
  True if A and B have the same meaning, False if they do not.
  """
  gpt4_prediction = predict_sample_openai_chatgpt(prompt)
  return gpt4_prediction['prediction']


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_excel(file_path):
    df = pd.read_excel(file_path+"/annotation.xlsx", sheet_name=1)
    id_explanations_dict = {}
    for index, row in df.iterrows():
        id = row['ID']
        explanations = [
            row['Crowd Explanations[0]'], 
            row['Crowd Explanations[1]'], 
            row['Crowd Explanations[2]'], 
            row['Crowd Explanations[3]'], 
            row['Crowd Explanations[4]']
        ]
        explanations = [explanation for explanation in explanations if pd.notna(explanation)]
        id_explanations_dict[id] = explanations
    
    df_storys = pd.read_excel(file_path+"/dataset.xlsx", sheet_name=2)
    id_story_dict = {}
    for index, row in df_storys.iterrows():
        id = row['ID']
        story = row['Knowledge background']
        id_story_dict[id] = story

    print(id_explanations_dict, id_story_dict)
    return id_explanations_dict, id_story_dict


PATH = {
    'BLIP2-XL': '../results/image_explanation/blip2-flan-t5-xl.json',
    'BLIP2-XXL': '../results/image_explanation/blip2-flan-t5-xxl.json',
    'InstructBLIP-XL': '../results/image_explanation/instructblip-flan-t5-xl.json',
    'InstructBLIP-XXL': '../results/image_explanation/instructblip-flan-t5-xxl.json',
    'mPLUG-owl-7B': '../results/image_explanation/mplug-owl-llama-7b.json',
    'mPLUG-owl2-7B': '../results/image_explanation/mplug-owl2-llama2-7b.json',
    'LLaVA-1.5-7B': '../results/image_explanation/llava-v1.5-7b.json',
    'LLaVA-1.6-7B': '../results/image_explanation/llava-v1.6-vicuna-7b.json',

    'MMICL-Z': '../results/image_explanation/MMICL-Instructblip-T5-xl-z.json',
    'MMICL-F': '../results/image_explanation/MMICL-Instructblip-T5-xl-f.json',
    'MMICL-C': '../results/image_explanation/MMICL-Instructblip-T5-xl-c.json',
    'Openflamingo-Z': '../results/image_explanation/OpenFlamingo-3B-vitl-mpt1b-z.json',
    'Openflamingo-F': '../results/image_explanation/OpenFlamingo-3B-vitl-mpt1b-f.json',
    'Openflamingo-C': '../results/image_explanation/OpenFlamingo-3B-vitl-mpt1b-c.json',
    'GPT-4V-Z': '../results/image_explanation/sk-hBAtTjpW4F6vEcyTB8De1577Af1d4cFf9d061502C56f390c-z.json',
    'GPT-4V-F': '../results/image_explanation/sk-hBAtTjpW4F6vEcyTB8De1577Af1d4cFf9d061502C56f390c-f.json',
    'GPT-4V-C': '../results/image_explanation/sk-hVCRy4zJf0gFSEar716c6397E0A74308A96305C46121FeEc-c.json',
    # pipeline GT
    'llama-2-7b-gt': '../results/pipeline_explanation/Llama-2-7b-z-n-no.json',
    'vicuna-1.5-7b-gt': '../results/pipeline_explanation/vicuna-7b-v1.5-z-n-no.json',
    'GPT-3.5-gt': '../results/pipeline_explanation/sk-6H72K8DB8HOw0u8u1768Cc8e1fEa438284Ee38C807A7E76a-z-n-no.json',
    # pipeline predict caption
    'llama-2-7b-pre': '../results/pipeline_explanation/Llama-2-7b-z-n-yes.json',
    'vicuna-1.5-7b-pre': '../results/pipeline_explanation/vicuna-7b-v1.5-z-n-yes.json',
    'GPT-3.5-pre': '../results/pipeline_explanation/sk-6H72K8DB8HOw0u8u1768Cc8e1fEa438284Ee38C807A7E76a-z-n-yes.json'


}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate predicted explanations against crowd explanations.')
    parser.add_argument("-model", type=str, required=True, choices=list(PATH.keys()))
    args = parser.parse_args()

 
    json_file_path = PATH[args.model]
    excel_file_path = '../dataset'

    predictions = load_json(json_file_path)
    references, storys = load_excel(excel_file_path)

    result_dict = {}
    for id, predicted_explanation in tqdm(predictions.items()):
        result_list = []
        flag=0
        for ture_explanation in references[id]:
            try:
                res = gpt4_estimetion(ture_explanation, predicted_explanation, storys[id])
                result_list.append(res)
                # print(res)
            except: 
                flag=1
                break
                
        if flag == 1:
            break
        result_dict[id] = result_list
        
        
        

    # output = gpt4_estimetion("This image does not conform to the story of the tortoise and the hare, because in the original story, the rabbit is sleeping while the tortoise is crawling, whereas in the image, both the tortoise and the rabbit are sleeping.",
    #  "This picture does not match the story of the tortoise and the hare racing. Because in the original story, the tortoise was racing hard while the hare was sleeping. In this picture, both the tortoise and the hare are sleeping.", 
    #  "The Tortoise and the Hare")

    # print(output)

    with open(f"./image_explanation/{args.model}"+".json", 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)

        

