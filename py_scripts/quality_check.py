import json
import os
import argparse
import sys
import re
from tqdm import tqdm
import asyncio
import numpy as np
from datetime import datetime
import argparse
sys.path.append('../')
from utils.open_llm_call import *
from utils.utils import *


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Input directory')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output directory')
    args = parser.parse_args()
    user_prompt = 'You will be given two prompts, one in Standard English and one in African American English. Determine whether the African American English prompt is a valid paraphrase of the Standard English prompt. Ignore the semantic validaty of the Standard English prompt.\nStandard English: "[SAE_PROMPT]"\nAfrican American English: "[AAVE_PROMPT]"\nIs the African American English prompt a valid paraphrase of the Standard English prompt?'
    client = get_client()
    responses = []
    data = json.load(open(args.input_dir, 'r'))
    print(f'Processing {args.input_dir}')
    file_name = args.input_dir.split('/')[-1]
    original_prompts = [ele['prompt'] for ele in data['vanilla']['original']]
    aave_prompts = [ele['prompt'] for ele in data['vanilla']['aave']]
    invalid_paras = []
    for i in tqdm(range(len(original_prompts))):
        curr_prompt = user_prompt.replace('[SAE_PROMPT]', original_prompts[i]).replace('[AAVE_PROMPT]', aave_prompts[i])
        response = await generate_from_openai_chat_completion(client,
                                                                model='gpt-4o-0513',
                                                                task_prompts=[curr_prompt],
                                                                n_choices=3,
                                                                temperature=0.7)
        responses.append(response)
        # save responses
        # mkdir if not exists
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(f'{args.output_dir}/{file_name.replace('.json', '_quality_checked_responses.json')}', 'w') as f:
            json.dump(responses, f)
        try:
            answers = [response[0]['choices'][n]['message']['content'].lower() for n in range(3)]
            # print index, original/aave prompt if at least two of the three responses are no
            yes_count=0
            for ans in answers:
                if 'yes,' in ans or 'yes.' in ans:
                    yes_count+=1
            if yes_count <= 1:
                print({i})
                print(f'Original: {original_prompts[i]}\n\nAAVE: {aave_prompts[i]}\n###')
                invalid_paras.append(f'Index {i}:\nOriginal: {original_prompts[i]}\n\nAAVE: {aave_prompts[i]}\n\nReason 1: {answers[0]}\nReason 2: {answers[1]}\nReason 3: {answers[2]}###')
        except:
            pass
    if invalid_paras:
        with open(f'../redial/redial_gold/invalid_paras/{file_name.replace('.json', 'invalid_paras.txt')}', 'w') as f:
            f.write('######\n\n'.join(invalid_paras))
