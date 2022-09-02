import base64
import requests
import os
import json
import time

url = "http://127.0.0.1:5000/infer"
image_folder_path = "images"
image_files = os.listdir(image_folder_path)

if __name__ == "__main__":
    st = time.time()
    i_ = 0
    for img in image_files:
        i_ += 1
        with open(os.path.join(image_folder_path, img), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        
        data = {}
        data['image_name'] = img
        data['image'] = encoded_string

        response = requests.post(url, data)
        print("response")
        response_dict = json.loads(response.text)
        for i in response_dict:
            print("key: ", i, "val: ", response_dict[i])
    
    print(i_, time.time() - st)
