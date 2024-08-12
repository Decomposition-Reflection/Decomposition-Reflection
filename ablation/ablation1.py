# coding=utf-8
import numpy as np
import os
import sys
import time
from evaluate import load
import urllib.request, json
import re
import argparse
from sent_similarity import get_scores
from self_bleu import calculate_selfBleu
from bert_score.utils import model2layers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

bertscore = load("bertscore/bertscore.py")

file_path = "data/domain_lfqa_test.json"
questions1 = []
references1 = []
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    for entry in data:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        questions1.append(question)
        references1.append(answer)


file_path = "data/rand_lfqa_test.json"
questions2 = []
references2 = []
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    for entry in data:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        questions2.append(question)
        references2.append(answer)

parser = argparse.ArgumentParser()
parser.add_argument('--reference_file_asqa', type=str, default='data/asqa_train_200.json', help='Reference file3.')
args, _ = parser.parse_known_args()
with open(args.reference_file_asqa, 'r', encoding='utf-8') as file:
    lines = file.readlines()
data3 = [json.loads(line) for line in lines]
references3 = [ex["answer"] for ex in data3]
questions3 = [ex["question"] for ex in data3]


parser = argparse.ArgumentParser()
parser.add_argument('--reference_file_eli5', type=str, default='data/eli5_test_200_new.json', help='Reference file4.')
args, _ = parser.parse_known_args()
with open(args.reference_file_eli5, 'r', encoding='utf-8') as file:
    lines = file.readlines()
data4 = [json.loads(line) for line in lines]
references4 = [ex["answer"] for ex in data4]
questions4 = [ex["question"] for ex in data4]

sys_prompt = "You are an all-around expert with deep insights in various fields."
with open("Prompt/DPprompt2.json", "r") as json_file:
    all_prompt = json.load(json_file)
decomposition_goal = all_prompt["decomposition_goal"]
decomposition_rule = all_prompt["decomposition_rule"]
decomposition_example = all_prompt["decomposition_example"]
decomposition_act = all_prompt["decomposition_act"]
prefix = all_prompt["prefix"]
merge_goal = all_prompt["merge_goal"]
merge_rule = all_prompt["merge_rule"]
merge_example = all_prompt["merge_example"]
merge_act = all_prompt["merge_act"]

def compute_bertscore(predictions, references):
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="facebook/roberta-large",
        num_layers=model2layers["roberta-large"],
    )
    return results['f1']

def calculate_rouge(predictions, references):
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
    return results


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', " ", text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def extract_answer(generated):
    if '\n' not in generated:
        last_line = generated
    else:
        last_line = generated.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]
    after_colon = after_colon.strip()
    if not after_colon.strip():
        return ""
    return normalize_answer(after_colon)

def parse_questions(question_string):
    questions = question_string.split('\n')
    print("questions",questions)
    sub_questions = []
    for question in questions:
        match = re.match(r'^Q\d+: (.+\?)$', question)
        if match:
            sub_question = match.group(1)
            sub_questions.append(sub_question)
            print("sub_question",sub_question)
    print("sub_questions",sub_questions)
    return sub_questions

def extract_questions(text):
    pattern = r'(Q\d+:.+?\?)'
    matches = re.findall(pattern, text)
    print("matches",matches)
    return matches

def check_answer_completeness(answer):
    answer = answer.rstrip()
    if answer.endswith(':') or answer.endswith(',') or not answer.endswith(('.', '!', '?')):
        return False
    else:
        return True


model_path = "model/Llama-2-7b-chat-hf"


def output_7b1(questions):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", offload_buffers=True)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    answers = []
    input_ids = tokenizer(questions, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.9)
    ans = tokenizer.decode(outputs[0])
    answers.append(ans)
    return answers


def output_7b2(questions):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", offload_buffers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    answers = []
    input_ids = tokenizer(questions, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.9)
    ans = tokenizer.decode(outputs[0])
    answers.append(ans)
    return answers


def remove_sequence(A, B):
    """
    Remove sequences in list B that are present in sequence A.

    Args:
    A (str): The sequence to be removed.
    B (list): List of sequences.

    Returns:
    list: List B with sequences from A removed.
    """
    sub_questions = A.split('\n')[1:-1]
    processed_B = []
    for sequence in B:
        if not any(sub_question in sequence for sub_question in sub_questions):
            processed_B.append(sequence)
    return processed_B

def get_top_sublists_and_removed_elements(arrage):
    sublists_and_scores = [(arrage[:i] + arrage[i+1:], calculate_selfBleu(arrage[:i] + arrage[i+1:]), arrage[i]) for i in range(len(arrage))]

    sublists_and_scores.sort(key=lambda x: x[1], reverse=True)

    top_10_sublists_and_removed_elements = sublists_and_scores[:10]

    removed_elements = [item[2] for item in top_10_sublists_and_removed_elements]

    return removed_elements


def renumber_and_merge_questions(questions):
    renumbered_questions = []
    for i, question in enumerate(questions, start=1):
        colon_index = question.index(':')
        new_question = f"Q{i}{question[colon_index:]}"
        renumbered_questions.append(new_question)
    result = ' '.join(renumbered_questions)
    return result

def renumber_questions(questions):
    renumbered_questions = []
    for i, question in enumerate(questions, start=1):
        colon_index = question.index(':')
        new_question = f"Q{i}{question[colon_index:]}"
        renumbered_questions.append(new_question)

    return renumbered_questions

rouge_results = []
for experiment in range(3):
    answers1 = []
    answers2 = []
    i = 0
    n = len(questions1)
    while i<n:
        prompt1 = decomposition_goal + '\n' + decomposition_rule + '\n' + decomposition_example + '\n' + "Main question:" + questions1[i] + '\n' + decomposition_act
        output1 = output_7b1(prompt1)
        x = len(prompt1)
        output1 = output1[0][x:-4]
        output1_arrays = extract_questions(output1)
        output1_arrays = get_top_sublists_and_removed_elements(output1_arrays)
        output1_arrays = renumber_questions(output1_arrays)
        output1 = renumber_and_merge_questions(output1_arrays)
        merged_output2 = ""
        merge_answer = ""
        for j, output1_array in enumerate(output1_arrays, 1):
            a_loop_key = 0
            prompt_prefix1 = "Please answer the question\n"
            prompt2 = f'Question: {output1_array}\n{prompt_prefix1}Answer:'
            max_score = 0
            max_output = ""
            prompt22 = prompt2
            while a_loop_key < 5:    
                a_loop_key += 1
                threshold_a = 0.75
                output2_array = output_7b2(prompt22)
                x = len(prompt22)  #remove
                output2_array = output2_array[0][x:-4]
                prompt22 = prompt2 + '\n' + output2_array
                score2 = get_scores(output1_array, [output2_array])
                
                if score2[0] >= threshold_a and check_answer_completeness(output2_array):
                    max_score = score2[0]
                    max_output = output2_array
                    break
                elif not check_answer_completeness(output2_array):
                    loop_prompt2 = "Your answer is not a complete sentence. Please generate a complete answer."
                else:
                    loop_prompt2 = f"The score for this question's response is {score2[0]}, which is lower than the threshold of {threshold_a}. This indicates that there is too much confusion regarding the answer to this question. Please refine the answer to improve the score."
                        
                if max_score < score2[0]:
                    max_score = score2[0]
                    max_output = output2_array
                prompt22 = prompt22 + '\n' + loop_prompt2
            subquestion = f"Q{j}:{output1_arrays[j-1]}\n"
            subanswer = f"A{j}:{max_output}\n\n"
            merged_output2 += subquestion
            merged_output2 += subanswer
    
        prompt3 = merge_goal + '\n' + merge_rule + '\n' + merge_example + '\n' + "Main question:" + questions1[i] + '\n' + merged_output2 + merge_act
        m_loop_key = 0
        max_score3 = 0
        max_output3 = ""
        prompt33 = prompt3
        while m_loop_key < 5:
            m_loop_key += 1
            output3 = output_7b1(prompt33)
            x = len(prompt33)  #remove
            output3 = output3[0][x:-4]
            if not merged_output2:
                max_score3 = -1024
                max_output3 = output3
                break
            prompt33 = prompt3 + '\n' + output3
            score3 = compute_bertscore([questions1[i]], [output3])
            threshold_m = 0.75
            
            if score3[0] >= threshold_m and check_answer_completeness(output3):       
                max_score3 = score3[0]
                max_output3 = output3
                break
            elif not check_answer_completeness(output3):
                loop_prompt3 = "Your answer is not a complete sentence. Please generate a complete answer."
            else:
                loop_prompt3 = f"The score for the final answer is {score3[0]}, which is lower than the threshold of {threshold_m}. This means the final answer does not meet our set of two rules. Please refine the final answer to improve its effectiveness score."
                    
            if max_score3 < score3[0] :
                max_score3 = score3[0]
                max_output3 = output3
            prompt33 = prompt33 + '\n' + loop_prompt3
        answers1.append(max_output3)
        answers2.append(merged_output2)
        i += 1
    
    
    with open(f'results/llama/ablation/experqa_loseloop1_answers{experiment+1}(0.750.75).json', 'w') as file:
            json.dump(answers1, file)
    with open(f'results/llama/ablation/QAdatabase/expertqa_loseloop1_answers{experiment+1}(0.750.75)_QAdatabase.json', 'w') as file:
            json.dump(answers2, file)
    print("expertqa domain rouge:")
    rouge_score = calculate_rouge(answers1, references1)
    rouge_results.append(rouge_score)

rouge1_scores = [result['rouge1'] for result in rouge_results]
rouge2_scores = [result['rouge2'] for result in rouge_results]
rougeL_scores = [result['rougeL'] for result in rouge_results]
rouge1_mean = np.mean(rouge1_scores)
rouge2_mean = np.mean(rouge2_scores)
rougeL_mean = np.mean(rougeL_scores)
rouge1_std = np.std(rouge1_scores)
rouge2_std = np.std(rouge2_scores)
rougeL_std = np.std(rougeL_scores)
print("Rouge-1 Mean:", rouge1_mean)
print("Rouge-1 Std Dev:", rouge1_std)
print("Rouge-2 Mean:", rouge2_mean)
print("Rouge-2 Std Dev:", rouge2_std)
print("Rouge-L Mean:", rougeL_mean)
print("Rouge-L Std Dev:", rougeL_std)