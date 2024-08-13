# -*- coding: utf-8 -*-
'''
Remember to export you API keys first.
export OPENAI_API_KEY='openai_api_key'
export SERPER_API_KEY='serper_api_key'  https://serper.dev/dashboard
'''

from factool import Factool
import json
import os
import time
import argparse

factool_instance = Factool("gpt-3.5-turbo")

file_path = "FactulityDP-results/expertqa/domain_lfqa_test.json"
prompts1 = []
references1 = []
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    for entry in data:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        prompts1.append(question)
        references1.append(answer)

parser = argparse.ArgumentParser()
parser.add_argument('--reference_file_asqa', type=str, default='FactulityDP-results/asqa_train_200.json', help='Reference file3.')
args, _ = parser.parse_known_args()
with open(args.reference_file_asqa, 'r', encoding='utf-8') as file:
    lines = file.readlines()
data3 = [json.loads(line) for line in lines]
references3 = [ex["answer"] for ex in data3]
prompts3 = [ex["question"] for ex in data3]

parser = argparse.ArgumentParser()
parser.add_argument('--reference_file_eli5', type=str, default='FactulityDP-results/eli5_test_200_new.json', help='Reference file4.')
args, _ = parser.parse_known_args()
with open(args.reference_file_eli5, 'r', encoding='utf-8') as file:
    lines = file.readlines()
data4 = [json.loads(line) for line in lines]
references4 = [ex["answer"] for ex in data4]
prompts4 = [ex["question"] for ex in data4]

with open("path/to/file/that/you/want/to/test", 'r', encoding='utf-8') as file_B:
    responses1 = json.load(file_B)

assert len(prompts1) == len(responses1), "length is right"

progress_file = 'path/to/file/that/you/want/to/test_progress1.json'
if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as pf:
        progress_data = json.load(pf)
        start_index = progress_data.get('last_processed_index', 0) + 1
else:
    start_index = 0

output_file = 'path/to/file/that/you/want/to/test_factool_answers1.json'
if os.path.exists(output_file) and start_index > 0:
    with open(output_file, 'r', encoding='utf-8') as file_C:
        results = json.load(file_C)
else:
    results = []

for index in range(start_index, len(prompts1)):
    input_item = {
        "prompt": prompts1[index].strip(),
        "response": responses1[index].strip(),
        "category": "kbqa"
    }

    try:
        response_list = factool_instance.run([input_item])
        print("response_list:",response_list)
        results.append(response_list)
    except Exception as e:
        print(f"Error processing item at index {index}, error: {e}")
        if "quota" in str(e).lower():
            print("Quota exceeded, waiting for 60 seconds before retrying...")
            time.sleep(10)
        continue

    with open(output_file, 'w', encoding='utf-8') as file_C:
        json.dump(results, file_C, ensure_ascii=False, indent=4)

    with open(progress_file, 'w', encoding='utf-8') as pf:
        json.dump({'last_processed_index': index}, pf)

print("Processing completed.")

