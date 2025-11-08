import json
from termcolor import colored

PATH = 'irish_deephermes_llama31_8b_translated_bespoke_processed_native_lang_cot_rep12_0_to_5--checkpoint-1338_lc2024_train_None_False_0_-1.json'

# for i in ['0.05', '0.2', '0.5', '2.0', '5.0', '20.0']:
# for i in ['20.0']:
#     PATH = f'irish_deepseek_llama31_8b_translated_bespoke_stratos_17k_filtered_weight_before_think_loss_{i}--checkpoint-1971_lc2024_train_None_False_0_-1.json'
print(f"Loading {PATH}...")
dct = json.load(open(PATH))

concepts_and_skills = 0
contexts_and_applications = 0
# count = 0
for subdct in dct.values():
    # if subdct['token_usages']['0.6'][0]['completion_tokens'] == 32768:
    #     count += 1
    # acc1 += subdct['responses']["0.6"][0]["correctness"]
    # acc2 += subdct['responses']["0.6"][1]["correctness"]
    if not subdct['responses']["0.6"][0]["correctness"]:
        print(subdct["question_id"])
        print(int(subdct["question_id"].split(' ')[3][1]))
        print(colored('Response: ', 'red'), subdct['responses']["0.6"][0]['content'][-100:])
        print(colored('Response: ', 'red'), subdct['responses']["0.6"][0]['pred'][-100:])
        print(colored('Answer: ', 'green'), subdct['responses']["0.6"][0]['answer'])
        print()
        print()
        print()
        if int(subdct["question_id"].split(' ')[3][1]) < 7:
            concepts_and_skills += 1
        else:
            contexts_and_applications += 1
        # count += 1

print(f"Concepts and Skills: {32 - concepts_and_skills}")
print(f"Contexts and Applications: {23 - contexts_and_applications}")
    # print((30 - count) / 30 * 100)


# for aime:
'''import json
from termcolor import colored
from collections import Counter
from skythought_evals.util.math_parsing_util import math_equal

PATH = 'irish_deepseek_llama31_8b_translated_bespoke_stratos_17k--checkpoint-1002_irish_aime_train_None_False_0_-1-normal-prompt.json'
PATH = 'irish_deepseek_llama31_8b_translated_bespoke_stratos_17k--checkpoint-1002_aime_train_None_False_0_-1.json'
# for i in ['0.05', '0.2', '0.5', '2.0', '5.0', '20.0']:
#     for ckpt in [657, 1314, 1971]:
# PATH = f'irish_deepseek_llama31_8b_translated_bespoke_stratos_17k_filtered_weight_before_think_loss_{i}--checkpoint-{ckpt}_aime_train_None_False_0_-1.json'

print(f"Loading {PATH}...")
dct = json.load(open(PATH))

majority_vote_acc = 0
for subdct in dct.values():
    votes = [pred['pred'] for pred in subdct['responses']["0.6"]]
    # voted_raw_prediction = max(set(votes), key=votes.count)
    vote_counts = Counter(votes)
    max_count = max(vote_counts.values())
    top_items = [item for item, count in vote_counts.items() if count == max_count]
    for item in top_items:
        if math_equal(item, subdct['responses']["0.6"][0]['answer']):
            majority_vote_acc += 1
            break

# print(f"Total Questions: {len(dct)}")
# print(f"Majority Vote Correct: {majority_vote_acc}")
print(f"Majority Vote Accuracy: {majority_vote_acc / len(dct) * 100:.2f}%")
'''
