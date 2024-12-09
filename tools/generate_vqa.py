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
    min_ms_between_api_calls: int = 10000,
    engine: str = "gpt-3.5-turbo",
    max_tokens: int = 100,
):
    parameters, time_of_last_api_call = set_openai_parameters(engine, max_tokens)
    parameters["prompt"] = prompt
    api_key = "sk-6H72K8DB8HOw0u8u1768Cc8e1fEa438284Ee38C807A7E76a"
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
    wait_time = 5
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

def gpt_estimetion(caption):
  prompt= f"""
    Please extract candidate answers from the given caption, 
    then generate questions based on the candidate answers, 
    and finally match the answers with the questions. Retain the matched pairs and discard the unmatched ones.
    Caption: {caption}
    Just return the results in the format answer: ..., question: ..., and present the output in JSON format.
  """
  gpt_prediction = predict_sample_openai_chatgpt(prompt)
  return gpt_prediction['prediction']


def load_excel(file_path):
    df = pd.read_excel(file_path,sheet_name=0)

    id_captions_dict = {}
    for index, row in df.iterrows():
        id = row['ID']
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


if __name__ == '__main__':
    excel_file_path = '../dataset/annotation.xlsx'
    id_captions = load_excel(excel_file_path)

    result_dict = {}

    for id, captions in tqdm(id_captions.items(), desc="Processing"):
        id_dict = {}
        for caption in captions:
            qa_json = gpt_estimetion(caption)
            try:
                qa_json = qa_json.split("```")
                qa_json = [qa for qa in qa_json if qa.strip().startswith("json")]
                qa_json = json.loads(qa_json[0].replace("json",""))
            except:
                print(id)
            id_dict[caption] = qa_json
        result_dict[id] = id_dict

    
    with open(f"../dataset/vqa.json", 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)



