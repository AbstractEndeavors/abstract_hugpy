from abstract_apis import *
from abstract_utilities import eatAll
def getUrl(*args):
    url = ""
    for i,arg in enumerate(args):
        if i == 0:
            url=arg
        else:
            url = f"{url}/{arg}"
    return url
url = "https://typicallyoutliers.com"
prefix = "hugpy/deepcoder"
endpoint = "deepcoder"
data={"prompt":"can you access the blinterblet?",
                      "max_new_tokens":500}
result = postRequest(getUrl(url,prefix,endpoint),data=data)
print(result.text)
