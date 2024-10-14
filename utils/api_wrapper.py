import os
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
import requests
import urllib.request
import ssl
import json
import time

class LlamaWrapper:
    
    def __init__(self, endpoint = None):
        self.allowSelfSignedHttps(True)

    def allowSelfSignedHttps(self, allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    def encode_data(self,
                    message,
                    temperature,
                    top_p,
                    max_new_tokens):
        data = {
        "input_data": {
            "input_string": message,
            "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
            }
            }
        }

        body = str.encode(json.dumps(data))
        return body

    def run(self,
            user_prompt,
            system_prompt='',
            model_version = 'llama3_8b',
            temperature = 0,
            top_p = 1,
            max_new_tokens = 4096):
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.model_dict[model_version]['api_key']), 'azureml-model-deployment': self.model_dict[model_version]['deployment'] }
        if system_prompt:
            message = [{'role': 'system', "content": system_prompt}, {'role': 'user', "content": user_prompt}]
        else:
            message = [{'role': 'user', "content": user_prompt}]
        body = self.encode_data(message=message, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        req = urllib.request.Request(self.model_dict[model_version]['url'], body, headers)
        # print(self.model_dict[model_version]['url'])
        status = 0
        while status < 3:
            try:
                # print(body)
                response = urllib.request.urlopen(req)
                result = response.read()
                # print(result)
                result = json.loads(result)
                # print(result)
                return result['output']
            except urllib.error.HTTPError as error:
                print("The request failed with status code: " + str(error.code))
                print(error.info())
                print(error.read().decode("utf8", 'ignore'))
                status += 1
                time.sleep(2)
        return "API NO RESPONSE"
    
    def all_models(self):
        return self.model_dict.keys()


# llama_client = LlamaWrapper()
# print(llama_client.post("Hello, how are you?",  model_version = 'mistralai-8x7b'))