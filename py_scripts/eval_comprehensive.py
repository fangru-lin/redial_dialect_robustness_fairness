import json
import sys
import asyncio
import argparse
import sys
import os
sys.path.append('../')
from utils.open_llm_call import *
from utils.utils import *


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        required=False,
                        help='batch size')
    parser.add_argument('--model_name',
                        type=str,
                        default='',
                        required=True,
                        help='Model name')
    parser.add_argument('--azure_endpoint',
                        type=str,
                        default="",
                        required=True,
                        help='Azure endpoint')
    parser.add_argument('--api_version',
                        type=str,
                        default=None,
                        required=True,
                        help='API version')
    parser.add_argument('--token_provider_scope',
                        type=str,
                        default=None,
                        required=True,
                        help='Token provider scope')
    parser.add_argument('--n_choices',
                        type=int,
                        default=1,
                        required=False,
                        help='Sample size')
    parser.add_argument('--temperature',
                        type=float,
                        default=0,
                        required=False,
                        help='Temperature')
    parser.add_argument('--sys_prompt',
                        type=str,
                        default='',
                        required=False,
                        help='System prompt')
    parser.add_argument('--input_path',
                        type=str,
                        default='',
                        required=True,
                        help='Input path')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output directory')
    parser.add_argument('--use_api',
                        type=bool,
                        default=False,
                        required=False,
                        help='Use API')
    parser.add_argument('--no_aave_test',
                        type=bool,
                        default=False,
                        required=False,
                        help='Disable AAVE test')
    parser.add_argument('--no_test_mode',
                        action='store_false',
                        dest='test_mode',
                        help='Disable test mode (default: True)')
    parser.add_argument('--no_cot',
                        type=bool,
                        default=False,
                        required=False,
                        help='Disable chain of thought')
    parser.add_argument('--sae_ablate',
                        type=bool,
                        default=False,
                        required=False,
                        help='SAE ablation')
    parser.add_argument('--url',
                        type=str,
                        default='',
                        required=False,
                        help='URL for open-source LLM if using API wrapper')
    parser.add_argument('--api_key',
                        type=str,
                        default='',
                        required=False,
                        help='API key for open-source LLM if using API wrapper')
    parser.add_argument('--deployment',
                        type=str,
                        default='',
                        required=False,
                        help='Deployment for open-source LLM if using API wrapper')
    args = parser.parse_args()
    # print all arguments
    print(args)
    if 'gpt' in args.model_name.lower():
        client = get_client(endpoint=args.azure_endpoint,)
    else:
        client = None
    with open(args.input_path, 'r') as f:
        comprehensive_dic = json.load(f)
    for dia in ['aave', 'original']:
        if args.no_aave_test and dia == 'aave':
            continue
        if dia == 'aave':
            transformed=True
        else:
            transformed=False
        for cot in [True, False]:
            if cot and args.no_cot:
                continue
            cot_cate = 'cot' if cot else 'vanilla'
            aave_cate = 'aave' if transformed else 'original'
            prompts = [ele['prompt'] for ele in comprehensive_dic['cot' if cot else 'vanilla'][dia]]
            gts = [ele['label'] for ele in comprehensive_dic['cot' if cot else 'vanilla'][dia]]
            
            if args.sae_ablate:
                prompts = [prompt+' Let\'s rephrase the question in Standard English first then answer it.' for prompt in prompts]

            await eval_redial(client,
                              prompts=prompts,
                              gts=gts,
                              task='comprehensive',
                              sys_prompt=args.sys_prompt,
                              use_api=args.use_api,
                              data_names=[ele['data_name'] for ele in comprehensive_dic[cot_cate][aave_cate]],
                              model_name=args.model_name,
                              output_dir=os.path.join(args.output_dir, 'comprehensive'),
                              cot=cot,
                              transformed=transformed,
                              batch_size=args.batch_size,
                              n_choices=args.n_choices,
                              temperature=args.temperature,
                              url=args.url,
                              api_key=args.api_key,
                              deployment=args.deployment)


if __name__ == "__main__":
    asyncio.run(main())
