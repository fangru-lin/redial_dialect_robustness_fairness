from openai import AsyncAzureOpenAI
from openai import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
    Timeout,
)
import asyncio
import json
import logging
import random
from azure.identity import AzureCliCredential, get_bearer_token_provider

ERROR_ERRORS_TO_MESSAGES = {
    BadRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    RateLimitError: "OpenAI API rate limit exceeded.",
    APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    APIError: "OpenAI API error: {e}",
}



def get_client(endpoint: str,
               api_version: str,
               token_provider_scope: str,
               token_provider_credential=AzureCliCredential(),):
    '''
    Get client for Azure OpenAI
        Parameters:
            endpoint (str): endpoint for Azure OpenAI
            api_version (str): api version for Azure OpenAI
            token_provider_credential: credential for token provider
            token_provider_scope (str): scope for token provider
        Returns:
            client (AsyncAzureOpenAI): client for Azure OpenAI
    '''
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(token_provider_credential, token_provider_scope),
        api_version=api_version,
    )
    return client


async def _throttled_openai_chat_completion_acreate(
    client,
    model: str,
    messages: list,
    temperature: float,
    top_p=1.0,
    n=3,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    initial_delay: float = 10,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,

):
    if not messages:
        return {}
    num_retries = 0
    delay = initial_delay
    for _ in range(max_retries):
        try:
            return await client.chat.completions.create(
                                                        model=model,
                                                        messages=messages,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        n=n,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop,
                                                        # max_tokens=100000
                                                    )
        except Exception as e:
            if isinstance(e, APIError):
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
            elif isinstance(e, BadRequestError):
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Invalid Request: Prompt was filtered"
                            }
                        }
                    ]
                }
            else:

                logging.warning(e)

            num_retries += 1
            print("num_retries=", num_retries)

            # Check if max retries has been reached
            if num_retries > max_retries:
                raise Exception(
                    f"Maximum number of retries ({max_retries}) exceeded."
                )

            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())
            print("new delay=", delay)
            await asyncio.sleep(delay)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        client,
        task_prompts: list[str],
        nshot_prompt: str = '',
        model: str = 'gpt-4o-0513',
        system_prompts: list[str] = [],
        n_choices: int = 1,
        temperature: float = 1.0,
) -> list:
    assert type(task_prompts) is list
    assert type(system_prompts) is list
    
    if system_prompts:
        messages = list()
        for i, task_prompt in enumerate(task_prompts):
            if not task_prompt:
                messages.append({})
                continue
            messages.append([{
                              'role': 'system',
                              'content': system_prompts[i]
                             },
                             {
                              "role": "user",
                              "content": nshot_prompt+task_prompt
                             }])
    else:
        messages = list()
        for task_prompt in task_prompts:
            # skip empty prompts
            if not task_prompt:
                messages.append({})
                continue
            messages.append([{
                              "role": "user",
                              "content": nshot_prompt+task_prompt,
                              }])
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=model,
            messages=message,
            temperature=temperature,
            top_p=1,
            n=n_choices,
            frequency_penalty=0,
            stop=None,
        ) for message in messages
    ]
    responses = await asyncio.gather(*async_responses)
    # reprompt if response is empty string by batching the same prompt again
    flag=False
    for retry in range(10):
        retry_queries = list()
        for i, response in enumerate(responses):
            return_message = json.loads(response.model_dump_json(indent=2))['choices'][0]['message']['content'] if type(response) != dict else ''
            if return_message == '':
                retry_queries.append((i, messages[i]))
                flag=True
        if flag:
            new_queries = [x[1] for x in retry_queries]
            new_async_responses = [
                                        _throttled_openai_chat_completion_acreate(
                                            client,
                                            model=model,
                                            messages=message,
                                            temperature=temperature,
                                            top_p=1,
                                            n=n_choices,
                                            frequency_penalty=0,
                                            stop=None
                                        ) for message in new_queries
                                    ]
            new_responses = await asyncio.gather(*new_async_responses)
            for i, new_response in enumerate(new_responses):
                responses[retry_queries[i][0]] = new_response
        if not flag:
            break
    # Note: will never be none because it's set, but mypy doesn't know that.
    # await openai.aiosession.get().close()  # type: ignore
    all_responses = [json.loads(x.model_dump_json(indent=2)) if type(x) is not dict else x for x in responses]
    return all_responses
