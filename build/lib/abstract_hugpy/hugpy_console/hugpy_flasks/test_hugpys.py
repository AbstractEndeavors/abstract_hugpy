from abstract_apis import *
from abstract_utilities import eatAll
code ='''def run_pruned_func(func, *args, **kwargs):
   args,kwargs = prune_inputs(func, *args, **kwargs)
   return func(*args, **kwargs)
'''
prompt=f"""Describe the purpose and functionality of the following code snippet.
Focus on its logic, data flow, and overall role within a system.
Identify key functions, inputs, and outputs, and avoid listing syntax or line-by-line commentary.

Code:
{code}
"""
def getUrl(*args):
    url = ""
    for i,arg in enumerate(args):
        if i == 0:
            url=arg
        else:
            url = f"{url}/{arg}"
    return url
url = "https://typicallyoutliers.com"
prefix = "hugpy"
endpoint = "zerosearch_generate"
data={"prompt":prompt,"max_new_tokens":200}
result = postRequest(getUrl(url,prefix,endpoint),data=data)
print(result)
