import transformers
import torch
from tqdm import tqdm
import os
import json
from .api_wrapper import *


def init_pipeline(model_id: str,
                  task: str='text-generation',
                  token: str='',
                  **kwargs):
    '''
    Initialize inference pipelines for different models.
        Parameters:
            model_id (str): The model ID to be used.
            task (str): The task to be performed by the model.
            token (str): The token to be used for the model.
        Returns:
            pipeline (transformers.Pipeline): The initialized pipeline.
            terminators (list): The list of terminators for the model.
    '''
    assert torch.cuda.is_available(), "This model needs a GPU to run ..."
    device = torch.cuda.current_device()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map="auto")
    # Initialize the pipeline
    if 'llama-3' in model_id.lower():
        terminators = [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        assert 'mistral' in model_id.lower() or 'gemma' in model_id.lower() or 'mixtral' in model_id.lower(), "Only Mistral/llama/gemma models are supported ..."
        terminators = [tokenizer.eos_token_id]

    tokenizer.pad_token_id = model.config.eos_token_id
    pipeline = transformers.pipeline(task=task,
                                     model=model,
                                     tokenizer=tokenizer,)
                                    #  device=device)
    print(f"Initialized pipeline for {model_id}.")
    # print(f"Initialized pipeline for {model_id} on device {device}.")
    return pipeline, terminators


def inference(pipeline: transformers.Pipeline,
              user_prompts: list,
              terminators: list,
              model: transformers.AutoModelForCausalLM=None,
              tokenizer: transformers.AutoTokenizer=None,
              save_dir: str=None,
              save_name: str=None,
              sys_prompts: list=[],
              max_new_tokens: int=4096,
              temperature: float=1,
              top_p: float=1.0,
              batch_size: int=1,
              num_return_sequences: int=1,
              instruction_tuned: bool=True,
              **kwargs):
    '''
    Wrap up the inference process.
        Parameters:
            pipeline (transformers.Pipeline): The initialized pipeline.
            user_prompt (str): The user prompt to be used.
            terminators (list): The list of terminators for the model.
            max_length (int): The maximum length of the output.
            sys_prompt (str): The system prompt to be used.
            temperature (float): The temperature to be used.
            do_sample (bool): Whether to sample the output.
            top_p (float): The top-p value to be used.
            batch_size (int): The batch size to be used.
            num_return_sequences (int): The number of sequences to be returned.
        Returns:
            outputs (list): The list of outputs from the model.
    '''
    assert isinstance(user_prompts, list), "The user prompts must be a list"
    assert isinstance(sys_prompts, list), "The system prompts must be a list"
    if temperature:
        do_sample = True
    else:
        do_sample = False
    if sys_prompts[0]:
        assert len(user_prompts) == len(sys_prompts), "The number of user prompts must be equal to the number of system prompts"
    try:
        model_name = pipeline.model.config._name_or_path
    except:
        model_name = model.config._name_or_path
    if instruction_tuned:
        if not sys_prompts[0]:
            batched_queries = [[{'role': 'user', 'content': user_prompts[i]}] for i in range(len(user_prompts))]
        else:
            if 'llama' in model_name.lower():
                batched_queries = [[ {"role": "system", "content": sys_prompts[i]}, {'role': 'user', 'content': user_prompts[i]}] for i in range(len(user_prompts))]
            else:
                assert 'mistral' in model_name.lower() or 'gemma' in model_name.lower() or 'mixtral' in model_name.lower(), "Only Mistral/llama/mixtral models are supported ..."
                # Mistral models do not require system prompts so append the user prompts to the system prompts
                batched_queries = [[{'role': 'user', 'content': sys_prompts[i]+'\n\n'+user_prompts[i]}] for i in range(len(user_prompts))]
    else:
        batched_queries = apply_template([[p] for p in user_prompts],model_name=model_name, urial='inst_1k_v4.help')
    try:
        with open(os.path.join(save_dir, save_name), 'r') as f:
            outputs = json.load(f)
    except:
        outputs = []
    start_idx = len(outputs)
    if save_dir:
        with open(os.path.join(save_dir, save_name.replace('output.json', 'input.json').replace('outputs.json', 'inputs.json')), 'w') as f:
            json.dump(batched_queries, f, indent=4)
    for i in tqdm(range(start_idx, len(batched_queries), batch_size)):
        if instruction_tuned:
            curr_outputs = pipeline(batched_queries[i:i+batch_size],
                                    max_new_tokens=max_new_tokens,
                                    eos_token_id=terminators,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                    top_p=top_p,
                                    return_full_text=False,
                                    batch_size=batch_size,
                                    num_return_sequences=num_return_sequences)
            outputs.extend([[oo["generated_text"] for oo in o] for o in curr_outputs])
        else:
            inputs = tokenizer(batched_queries[i:i+batch_size],
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=max_new_tokens)
            inputs.to('cuda')
            input_len = inputs['input_ids'].shape[1]
            output_ids = model.generate(**inputs,
                                        max_new_tokens=200,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_p=top_p,
                                        num_return_sequences=num_return_sequences)
            curr_outputs = [tokenizer.decode(output_id[input_len:], skip_special_tokens=True) for output_id in output_ids]
            outputs.extend(curr_outputs)
        if save_dir:
            with open(os.path.join(save_dir, save_name), 'w') as f:
                json.dump(outputs, f, indent=4)
    return [o[0] for o in outputs]


def pipe_and_infer(model_id: str,
                   save_dir: str=None,
                   save_name: str=None,
                   use_api: bool=False,
                   top_p: float=1.0,
                   max_new_tokens: int=4096,
                   **kwargs):
    '''
    Initialize and run inference on a model.
        Parameters:
            model_id (str): The model ID to be used.
            save_dir (str): The directory to save the outputs.
            save_name (str): The name of the file to save the outputs.
            use_api (bool): Whether to use the API for inference
            **kwargs: Additional keyword arguments for the inference and pipeline function.
        Returns:
            outputs (list): The list of outputs from the model.
    '''
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if not use_api:
        if 'it' in model_id.lower() or 'inst' in model_id.lower() or 'chat' in model_id.lower():
            kwargs['instruction_tuned'] = True
            pipeline, terminators = init_pipeline(model_id=model_id,
                                            **kwargs)
            outputs = inference(pipeline=pipeline,
                                terminators=terminators,
                                save_dir=save_dir,
                                save_name=save_name,
                                **kwargs)
            # unload the model
            pipeline.model = None
        else:
            kwargs['instruction_tuned'] = False
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, token=kwargs['token'], device_map="auto")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left', **kwargs)
            tokenizer.pad_token_id = model.config.eos_token_id
            outputs = inference(model=model,
                                pipeline=None,
                                terminators=None,
                                tokenizer=tokenizer,
                                save_dir=save_dir,
                                save_name=save_name,
                                **kwargs)
            # unload the model
            model = None
            
    else:
        try:
            with open(os.path.join(save_dir, save_name), 'r') as f:
                outputs = json.load(f)
        except:
            outputs = []
        start_idx = len(outputs)
        llama_client = LlamaWrapper(url=kwargs['url'],
                                    api_key=kwargs['api_key'],
                                    deployment=kwargs['deployment'])
        if kwargs['sys_prompts']:
            assert len(kwargs['user_prompts']) == len(kwargs['sys_prompts']), "The number of user prompts must be equal to the number of system prompts"
        for i in tqdm(range(len(kwargs['user_prompts']))):
            if i < start_idx:
                continue
            user_prompt = kwargs['user_prompts'][i]
            output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
                                      user_prompt=user_prompt,
                                      model_version=model_id,
                                      temperature=kwargs['temperature'],
                                      top_p=top_p,
                                      max_new_tokens=max_new_tokens)
            outputs.append(output)
            with open(os.path.join(save_dir, save_name), 'w') as f:
                json.dump(outputs, f, indent=4)
        # check no response
        flag = True
        count = 0
        while flag and count < 3:
            count+=1
            print("Checking for API NO RESPONSE ...")
            flag = False
            for i in tqdm(range(len(outputs))):
                if outputs[i] == "API NO RESPONSE":
                    # try to rerun the API
                    user_prompt = kwargs['user_prompts'][i]
                    output = llama_client.run(system_prompt = kwargs['sys_prompts'][i],
                                            user_prompt=user_prompt,
                                            model_version=model_id,
                                            temperature=kwargs['temperature'],
                                            top_p=top_p,
                                            max_new_tokens=max_new_tokens)
                    outputs[i] = output
                    with open(os.path.join(save_dir, save_name), 'w') as f:
                        json.dump(outputs, f, indent=4)
                    flag = True
        
    return outputs
