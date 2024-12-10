from datetime import timedelta
import datetime as dt
import json
import os
import numpy as np
from evalplus.data import write_jsonl
import re
from utils.llm_call import *
from tqdm import tqdm
from utils.open_llm_call import *


def find_answer_comprehensive(answer):
    '''
    Find the answer from the response.
        Parameters:
            answer (str): response
    '''
    try:
        ans = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)[-1].strip().lower()
        return ans
    except:
        return ''


def str_to_timedelta_list(time_str: str) -> list:
    '''
    Convert a string representation of a list of timedelta objects back to a list of timedelta objects.
        Parameters:
            time_str (str): time string in the format '[datetime.timedelta(seconds=4800), datetime.timedelta(seconds=4800)]'
        Returns:
            time_delta_list (list): list of timedelta objects
    '''
    # Evaluate the string within a controlled environment
    time_delta_list = eval(time_str, {"timedelta": timedelta, "datetime": dt})
    return time_delta_list


def check_correctness_comprehensive(parsed_response: str,
                                    gt: str):
    '''
    Check the correctness of the response.
        Parameters:
            parsed_response (str): parsed response
            gt (str): ground truth
        Returns:
            correctness (bool): whether the response is correct or not
    '''
    if not parsed_response:
        return False
    try:
        time_gt = str_to_timedelta_list(gt)
        return measure_perf(parsed_response, time_gt)[1]
    except:
        return False

def text_to_number_updated(sentence: str):
    # Updated mapping of number words to their numerical equivalents
    num_words = {
        "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 'a': 1,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000
    }

    # Helper function to convert a textual number expression to a number
    def text_number_to_num(text_num: str):
        parts = text_num.split()
        if 'and' in parts:
            parts.remove('and')

        total = 0
        current = 0

        for part in parts:
            if part in num_words:
                scale = num_words[part]
                if scale > 100:
                    current *= scale
                    total += current
                    current = 0
                elif scale == 100:
                    current *= scale
                else:
                    if current == 0:
                        current = scale
                    else:
                        current += scale
            else:
                # In case of numbers like 'forty-five'
                nums = part.split('-')
                for num in nums:
                    current += num_words.get(num, 0)

        return total + current

    # Regular expression pattern for matching text number expressions
    num_pattern = re.compile(r'\b(?:[a-zA-Z]+(?:-)?)+\b')

    # Find all matches
    matches = re.findall(num_pattern, sentence)

    # Process each match
    captured_patterns = {}
    for match in matches:
        number = text_number_to_num(match)
        if number > 0:
            captured_patterns[match] = number
            sentence = sentence.replace(match, str(number), 1)

    return sentence, captured_patterns


def measure_perf(response: str,
                 gold_timedelta: timedelta):
    text_num_set = set(["an", "one", "two", "three", "four", "five", 'a',
        "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "million", "half"])
    if not response:
        return [timedelta(), timedelta()], False
    potential_answers = list()
    try:
        # if a model follows instruction
        # answer should be in double quotes, and either the first or the last one is the answer
        potential_answers = [response.lower()]
        for i, answer in enumerate(potential_answers):
            if re.findall(r'\b\w+ and a half', answer):
                pattern = re.findall(r'\b\w+ and a half', answer)[0]
                prec_word = re.findall(r'\b\w+', pattern)[0]
                if prec_word not in text_num_set:
                    answer = answer.replace(pattern, f'and half {prec_word}')

                answer = answer.replace('half a ', '0.5')
                answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
                potential_answers[i] = answer
    except Exception as e:
        # if a model does not follow instruction
        # try to get response after 'is'
        try:
            answer = response.split('is ')[-1].lower().split('\n')[0]
        except:
            answer = response.lower()
        if re.findall(r'\b\w+ and a half', answer):
            pattern = re.findall(r'\b\w+ and a half', answer)[0]
            prec_word = re.findall(r'\b\w+', pattern)[0]
            if prec_word not in text_num_set:
                answer = answer.replace(pattern, f'and half {prec_word}')

            answer = answer.replace('half a ', '0.5')
            answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
            potential_answers = [answer]

    if not potential_answers:
        return [timedelta(), timedelta()], False
    for answer in potential_answers:
        if '=' in answer:
            answer = answer.split('=')[-1]
        if ' or ' in answer:
            answer = answer.split(' or ')[-1]
        if '(' in answer:
            answer = answer.split('(')[0]
        timedelta_ans = [timedelta(), timedelta()]
        if ' to ' in answer:
            return [timedelta(), timedelta()], False
        try:
            time_spans = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years|s|h|m|d|w)', answer)
            for time_span in time_spans:
                time = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?', time_span)[-1].replace(',','')
                unit = re.findall(r'\b[a-z]+', time_span)[-1].strip()
                if unit in ['year', 'years', 'y']:
                    delta = [timedelta(days=float(time)*365), timedelta(days=float(time)*366)]
                elif unit in ['month', 'months', 'm']:
                    # define a loose range for month
                    # match other units in same format
                    delta = [timedelta(days=float(time)*28), timedelta(days=float(time)*31)]
                elif unit in ['week', 'weeks', 'w']:
                    delta = [timedelta(weeks=float(time)), timedelta(weeks=float(time))]
                elif unit in ['day', 'days', 'd']:
                    delta = [timedelta(days=float(time)), timedelta(days=float(time))]
                elif unit in ['hour', 'hours', 'h']:
                    delta = [timedelta(hours=float(time)), timedelta(hours=float(time))]
                elif unit in ['minute', 'min', 'minutes', 'mins']:
                    delta = [timedelta(minutes=float(time)), timedelta(minutes=float(time))]
                elif unit in ['second', 'sec', 'seconds', 'secs']:
                    delta = [timedelta(seconds=float(time)), timedelta(seconds=float(time))]
                else:
                    raise ValueError(f'unit not found: {time_span}')
                timedelta_ans[0] += delta[0]
                timedelta_ans[1] += delta[1]

            if gold_timedelta[0] <= timedelta_ans[0] <= gold_timedelta[1]:
                return timedelta_ans, True
        except Exception:
            continue
        if gold_timedelta[0] <= timedelta_ans[1] <= gold_timedelta[1]:
            return timedelta_ans, True

    return [timedelta(), timedelta()], False


def find_answer_logic(answer,
                      data_name='folio'):
    '''
    Find the answer from the response.
        Parameters:
            answer (str): response
    '''
    if not answer:
        return None
    if data_name == 'folio':
        try:
            answer = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)[-1].strip().lower()
            return answer.strip()
        except:
            # the answer pattern is quite complicated
            # catch the final answer if the model does not follow the instruction
            answer = answer.lower()
            try:
                true_idx = re.search('necessarily true', answer, re.DOTALL).end()
            except:
                true_idx = None
            try:
                false_idx = re.search('necessarily false', answer, re.DOTALL).end()
            except:
                false_idx = None
            try:
                neither_idx = re.search('neither', answer, re.DOTALL).end()
            except:
                neither_idx = None
            
            if true_idx == None and false_idx == None and neither_idx == None:
                return None
            
            # select the idx with the largest value excluding None
            idxes = [true_idx, false_idx, neither_idx]
            for i in range(len(idxes)):
                if idxes[i] == None:
                    idxes[i] = -1
            idx = np.argmax(idxes)
            
            return ['necessarily true', 'necessarily false', 'neither'][idx]
    else:
        try:
            ans_span = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)[-1]
            return ans_span.strip()
        except:
            return ''


def check_correctness_logic(parsed_response,
                            gt):
    '''
    Check the correctness of the response.
        Parameters:
            parsed_response (str): response
            gt (str): ground truth
            data_name (str): dataset name
        Returns:
            correctness (bool): whether the response is correct or not
    '''

    if not parsed_response:
        return False
    try:
        return parsed_response.lower() == gt.lower()
    except:
        return False


def find_number(answer: str):
    '''
    Find number in the answer, including float numbers
        Parameters:
            answer: str, the answer to be processed
        Returns:
            numbers: float, the number in the answer
    '''
    try:
        pattern = r'<answer>(.*?)</answer>'
        match = re.findall(pattern, answer, re.DOTALL)[-1].replace(',','')
        number_pattern = r'\d+\.\d+|\d+'

        number = re.findall(number_pattern, match)[0]
        
        return number
    except:
        try:
            pattern = r'boxed{(.*?)}'
            match = re.findall(pattern, answer, re.DOTALL)[0].replace(',','')
            number_pattern = r'\d+\.\d+|\d+'
            number = re.findall(number_pattern, match)[0]
            return number
        except:
            return ''


def check_correctness_math(span: str,
                           gt,
                           data_name: str):
    '''
    Check the correctness of the answer for the mathematics dataset
        Parameters:
            ans: str: answer
            gt: ground truth, float or str
            data_name: str: dataset name
        Returns:
            int: correctness
    '''

    gt = gt.replace(',','')
    try:
        if type(gt) == str:
            if float(span) == float(re.findall(r'\d+\.\d+|\d+', gt)[0]):
                return 1
            return 0
        else:
            if float(span) == gt:
                return 1
            return 0
    except:
        return 0


def decode_rephrased_gen(response: str,
                         func_name: str):
    '''
    Decode the rephrased generation
        Parameters:
            response (str): rephrased generation
            func_name (str): function name
        Returns:
            decoded_response (str): decoded response
    '''
    try:
        decoded_response = response.replace('python_function', func_name)
        return decoded_response
    except:
        return None


def format_save_code_to_eval(outputs: list[dict],
                             function_names: list[str],
                             task_ids: list[str],
                             save_path: str,):
    outputs_to_eval = [{'task_id':task_ids[j], 'solution':decode_rephrased_gen(output['parsed_answer'], function_names[j])} for j, output in enumerate(outputs)]
    full_outputs_to_eval = [{'task_id':task_ids[j], 'solution':decode_rephrased_gen(output['response'], function_names[j])} for j, output in enumerate(outputs)]
    write_jsonl(save_path.replace('.json', f'_humaneval_to_eval.jsonl'), outputs_to_eval[:164])
    write_jsonl(save_path.replace('.json', f'_humaneval_to_eval_unparsed.jsonl'), full_outputs_to_eval[:164])
    write_jsonl(save_path.replace('.json', f'_mbpp_to_eval.jsonl'), outputs_to_eval[164:])
    write_jsonl(save_path.replace('.json', f'_mbpp_to_eval_unparsed.jsonl'), full_outputs_to_eval[164:])


def clean_code(answer):
    try:
        ans = answer.split('```')[1].strip()
        if ans.startswith('python\n'):
            ans = ans[7:]
        return ans.strip()
    except:
        return answer.strip()


def find_answer(response: str,
                data_name: str,
                task: str):
    '''
    Find the answer from the response.
        Parameters:
            response (str): response
            data_name (str): dataset name
            task (str): task name
        Returns:
            str: answer
    '''
    if not response:
        return ''
    if task == 'logic':
        return find_answer_logic(response, data_name=data_name)
    elif task == 'comprehensive':
        return find_answer_comprehensive(response)
    elif task == 'math':
        return find_number(response)
    elif task == 'algorithm':
        return clean_code(response)
    else:
        raise ValueError(f"Unknown task name: {task}")


def check_correctness(parsed_response: str,
                      gt: str,
                      task: str,
                      data_name: str=None):
    '''
    Check the correctness of the response.
        Parameters:
            parsed_response (str): parsed response
            gt (str): ground truth
            data_name (str): dataset name
            task (str): task name
        Returns:
            bool: whether the response is correct or not
    '''
    if task == 'logic':
        return check_correctness_logic(parsed_response, gt)
    elif task == 'comprehensive':
        return check_correctness_comprehensive(parsed_response, gt)
    elif task == 'math':
        return check_correctness_math(parsed_response, gt, data_name=data_name)
    elif task == 'algorithm':
        return None
    else:
        raise ValueError(f"Unknown task name: {task}")


def eval_save_ans(answers: list[list[str]],
                  prompts: list[str],
                  gts: list[str]=None,
                  original_gts: list[str]=None,
                  data_names: list=None,
                  save_path: str=None,
                  curr_idx: int=0,
                  task: str='logic',
                  function_names: list[str]=None,
                  task_ids: list[str]=None):
    '''
    Evaluate and save answers.
        Parameters:
            answers (list[list[str]]): List of answers.
            prompts (list[str]): List of prompts.
            gts (list[str]): List of ground truths.
            save_path (str): Path to save the dictionary.
            data_name (str): Name of the dataset.
        Returns:
            list: list of dictionaries of answers and evaluation results.
    '''
    if task == 'math':
        assert original_gts != None, "Original ground truth is required for math tasks"
    else:
        assert original_gts == None, "Original ground truth is not required for non-math tasks"
    if task != 'algorithm':
        assert gts != None, "Ground truth is required for non-coding tasks"
    else:
        assert gts == None, "Ground truth is not required for coding tasks"

    outputs = list()
    for i, answer in enumerate(answers):
        if type(answer) == str or not answer:
            parsed_anwers = find_answer(answer, task=task, data_name=data_names[i] if data_names else None)
        else:
            parsed_anwers = [find_answer(a, task=task, data_name=data_names[i] if data_names else None) for a in answer]
        if task != 'algorithm':
            ele = {"task_id":i+curr_idx,
                    "question":prompts[i],
                    "response":answer,
                    "parsed_ground_truth": gts[i], 
                    "parsed_answer": parsed_anwers,
                    "timestamp": str(datetime.now())}
            if type(parsed_anwers) == list:
                ele['correctness'] = [check_correctness(a, gts[i], task=task, data_name=data_names[i] if data_names else None) for a in parsed_anwers]
            else:
                ele['correctness'] = check_correctness(parsed_anwers, gts[i], task=task, data_name=data_names[i] if data_names else None)
        else:
            ele = {"task_id":i+curr_idx,
                    "question":prompts[i],
                    "response":answer,
                    "parsed_answer": parsed_anwers,
                    "timestamp": str(datetime.now())}
        if original_gts:
            ele['original_ground_truth'] = original_gts[i]
        outputs.append(ele)
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(outputs, f, indent=4)
        if task == 'algorithm':
            format_save_code_to_eval(outputs=outputs,
                                     function_names=function_names,
                                     save_path=save_path,
                                     task_ids=task_ids)
    return outputs


def mk_and_read_files_gpt(transformed,
                          cot,
                          output_dir,
                          model_name,
                          temperature,
                          task='logic'):
    '''
    Make and read files for GPT models.
        Parameters:
            data_name (str): Name of the dataset.
            transformed (bool): Whether the data is transformed.
            cot (bool): Whether the data is COT.
            output_dir (str): Output directory.
            model_name (str): Name of the model.
            temperature (float): Temperature.
            n_choices (int): Number of choices.
        Returns:
            str: Output path.
            list: List of outputs.
            list: List of logs.
            int: Start index.
            str: Output directory.
            str: Log name.
    '''
    out_name = f"aave_{transformed}_cot_{cot}.json"
    log_name = f"aave_{transformed}_cot_{cot}_log.json"
    out_dir = os.path.join(output_dir, model_name, f'temperature_{temperature}')
    out_path = os.path.join(output_dir, model_name, f'temperature_{temperature}', out_name)
    # mkdir if not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            outputs = json.load(f)
            start_idx = len(outputs)
            logs = json.load(open(os.path.join(out_dir, log_name), 'r'))
        print(f'Loaded {len(outputs)} {task} generations from {out_path}')
    else:
        start_idx = 0
        outputs = list()
        logs = list()
    
    return out_path, outputs, logs, start_idx, out_dir, log_name


def calc_acc(res):
    '''
    Calculate the accuracy.
        Parameters:
            res (list): List of results.
        Returns:
            float: Accuracy
    '''
    return round(np.mean([ele['correctness'] for ele in res]), 3)


async def eval_general_gpt(client,
                            prompts,
                            gts,
                            task,
                            sys_prompt,
                            data_names,
                            original_gts=None,
                            model_name='gpt-4o-0513',
                            output_dir = None,
                            cot = True,
                            transformed=False,
                            batch_size=2,
                            n_choices=1,
                            temperature=0,
                            function_names=None,
                            task_ids=None
                            ):
    '''
    Evaluate a GPT model.
        Parameters:
            client: OpenAI API client.
            prompts (list[str]): List of prompts.
            gts (list[str]): List of ground truths.
            task (str): Task name.
            data_name (str): Name of the dataset.
            model_name (str): Name of the model.
            output_dir (str): Output directory.
            cot (bool): Whether the data is COT.
            transformed (bool): Whether the data is transformed.
            batch_size (int): Batch size.
            n_choices (int): Number of choices.
            temperature (float): Temperature.
        Returns:
            None
    '''
    print(f'Generating results for {task} data')
    print(f'model_name: {model_name}')
    print(f'cot: {cot}')

    out_path, outputs, logs, start_idx, out_dir, log_name = mk_and_read_files_gpt(transformed=transformed,
                                                                                    cot=cot,
                                                                                    output_dir=output_dir,
                                                                                    model_name=model_name,
                                                                                    temperature=temperature,
                                                                                    task=task)

    for i in tqdm(range(start_idx, len(prompts), batch_size)):
        # generate answer for original prompt
        curr_prompts = prompts[i:i+min(batch_size, len(prompts)-i)]
        curr_responses = await generate_from_openai_chat_completion(client,
                                                                    model=model_name,
                                                                    task_prompts=curr_prompts,
                                                                    n_choices=n_choices,
                                                                    temperature=temperature)
        logs.extend([{'prompt': curr_prompts[ii], 'response': curr_responses[ii], 'system_prompt': sys_prompt} for ii in range(len(curr_prompts))])
        try:
            answers = [[response['choices'][ir]['message']['content'] for ir in range(len(response['choices']))] for response in curr_responses]
        except:
            answers = ['']
        curr_idx = i
        outputs.extend(eval_save_ans(answers=answers,
                                     prompts=curr_prompts,
                                     gts=gts[i:i+len(curr_prompts)] if gts else None,
                                     original_gts=original_gts[i:i+len(curr_prompts)] if original_gts else None,
                                     data_names=data_names[i:i+len(curr_prompts)],
                                     curr_idx=curr_idx,
                                     task=task,
                                     task_ids=task_ids[i+curr_idx:i+curr_idx+len(curr_prompts)] if task_ids else None,))
        with open(out_path, 'w') as f:
            json.dump(outputs, f, indent=4)
        with open(os.path.join(out_dir, log_name), 'w') as f:
            json.dump(logs, f, indent=4)

    print(f'{task} generations written to {out_path}')


def eval_general_open_llm(prompts,
                          gts,
                          task,
                          original_gts,
                          sys_prompt,
                          use_api,
                          data_names,
                          model_name='llama3_8b_instruct',
                          output_dir = None,
                          cot = True,
                          transformed=False,
                          batch_size=1,
                          n_choices=1,
                          temperature=0,
                          function_names=None,
                          task_ids=None,
                          url='',
                          api_key='',
                          deployment=''):
    '''
    Evaluate open LLM models.
        Parameters:
            prompts (list[str]): List of prompts.
            gts (list[str]): List of ground truths.
            task (str): Task name.
            sys_prompt (str): System prompt.
            use_api (bool): Whether to use the API.
            data_name (str): Name of the dataset.
            model_name (str): Name of the model.
            output_dir (str): Output directory.
            cot (bool): Whether the data is COT.
            transformed (bool): Whether the data is transformed.
            batch_size (int): Batch size.
            n_choices (int): Number of choices.
            temperature (float): Temperature.
            function_names (list[str]): List of function names.
            task_ids (list[str]): List of task IDs.
            url (str): URL for the API.
            api_key (str): API key.
            deployment (str): Deployment for the API.
        Returns:
            None
            
    '''
    full_outputs = list()
    for n in range(n_choices):
        save_dir = os.path.join(output_dir, model_name.split('/')[-1], f'temperature_{temperature}', f'nchoices_{n}')
        answers = pipe_and_infer(user_prompts=prompts,
                                model_id=model_name,
                                temperature=temperature,
                                sys_prompts=[sys_prompt for _ in range(len(prompts))],
                                save_dir=save_dir,
                                save_name=f'aave_{transformed}_cot_{cot}_outputs.json',
                                use_api=use_api,
                                batch_size=batch_size,
                                url=url,
                                api_key=api_key,
                                deployment=deployment)
        save_path = os.path.join(save_dir,
                                f'aave_{transformed}_cot_{cot}.json')
        outputs = eval_save_ans(answers=answers,
                                prompts=prompts,
                                function_names=function_names,
                                gts=gts if gts else None,
                                original_gts=original_gts if original_gts else None,
                                data_names=data_names,
                                save_path=save_path,
                                curr_idx=0,
                                task=task,
                                task_ids=task_ids if task_ids else None)
        if task != 'algorithm':
            split_by_entries = ['response', 'parsed_answer', 'correctness']
        else:
            split_by_entries = ['response', 'parsed_answer']
        if not full_outputs:
            for i in range(len(outputs)):
                full_outputs.append(dict())
                for entry in outputs[i].keys():
                    if entry in split_by_entries:
                        full_outputs[i][entry] = [outputs[i][entry]]
                    else:
                        full_outputs[i][entry] = outputs[i][entry]
        else:
            for i in range(len(outputs)):
                for entry in split_by_entries:
                    full_outputs[i][entry].append(outputs[i][entry])

    print(f'{task} generations written to {save_path}')


async def eval_redial(client,
                    prompts: list[str],
                    task: str,
                    gts: list=None,
                    sys_prompt: str='',
                    original_gts: list=None,
                    use_api: bool=True,
                    data_names: list=None,
                    model_name: str='gpt-4o-0513',
                    output_dir: str= None,
                    cot: bool=True,
                    transformed: bool=False,
                    batch_size=1,
                    n_choices=1,
                    temperature=0,
                    function_names=None,
                    task_ids=None,
                    url='',
                    api_key='',
                    deployment=''):
    '''
    Evaluate any model.
        Parameters:
            client: OpenAI API client.
            prompts (list[str]): List of prompts.
            gts (list[str]): List of ground truths.
            task (str): Task name.
            sys_prompt (str): System prompt.
            use_api (bool): Whether to use the API.
            data_name (str): Name of the dataset.
            model_name (str): Name of the model.
            output_dir (str): Output directory.
            cot (bool): Whether the data is COT.
            transformed (bool): Whether the data is transformed.
            batch_size (int): Batch size.
            n_choices (int): Number of choices.
            temperature (float): Temperature.
            function_names (list[str]): List of function names.
            task_ids (list[str]): List of task IDs.
            url (str): URL for the API.
            api_key (str): API key.
            deployment (str): Deployment for the API.
            
        Returns:
            None
    '''
    print(f'Evaluating {task} data')
    print(f'model_name: {model_name}')
    print(f'cot: {cot}')
    print(f'AAVE: {transformed}')
    if 'gpt' in model_name:
        await eval_general_gpt(client=client,
                                prompts=prompts,
                                gts=gts,
                                original_gts=original_gts,
                                function_names=function_names,
                                task=task,
                                sys_prompt=sys_prompt,
                                data_names=data_names,
                                model_name=model_name,
                                output_dir=output_dir,
                                cot=cot,
                                transformed=transformed,
                                batch_size=batch_size,
                                n_choices=n_choices,
                                temperature=temperature,
                                task_ids=task_ids)
    else:
        eval_general_open_llm(prompts=prompts,
                                gts=gts,
                                original_gts=original_gts,
                                task=task,
                                sys_prompt=sys_prompt,
                                function_names=function_names,
                                use_api=use_api,
                                data_names=data_names,
                                model_name=model_name,
                                output_dir=output_dir,
                                cot=cot,
                                transformed=transformed,
                                batch_size=batch_size,
                                n_choices=n_choices,
                                temperature=temperature,
                                task_ids=task_ids,
                                url=url,
                                api_key=api_key,
                                deployment=deployment)
        