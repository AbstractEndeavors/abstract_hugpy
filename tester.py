from abstract_apis import *
from abstract_apis.request_utils import *
import logging,os
# Suppress logs below WARNING level
def getDownloads(url, params, headers, output_file):
    """Send a GET request and save the response content to a file."""
    try:
        response = requests.get(url, params=params, headers=headers, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved file: {output_file}")
        return {"status": "success", "file": output_file}
    except requests.RequestException as e:
        print(f"GET request failed: {e}")
        return {"error": str(e)}

client_id = "id_rsa_small_rat"
url = "https://clownworld.biz/media/generate_client_keys"
resp = postRequest(url,data={"client_id":client_id},headers=get_headers())
url = "https://clownworld.biz/media/download_client_key"
# Step 2: Download key files
url = "https://clownworld.biz/media/download_client_key"
key_types = [
    ("ppk", f"{client_id}_id_rsa.ppk"),
    ("private", f"{client_id}_id_rsa"),
    ("public", f"{client_id}_id_rsa.pub")
]

for key_type, output_file in key_types:
    params = {"client_id": client_id, "key_type": key_type}
    result = getDownloads(url, params, get_headers(), output_file)
    if "error" in result:
        print(f"Failed to download {key_type} key: {result['error']}")
print(resp)
