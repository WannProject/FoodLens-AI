import requests
import json

url = "https://wanndev14-yolo-api.hf.space/detect-gizi"

try:
    with open("bakso.jpg", "rb") as f: 
        files = {"image": f}
        response = requests.post(url, files=files, timeout=30)
    
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
    
except Exception as e:
    print("Error:", e)