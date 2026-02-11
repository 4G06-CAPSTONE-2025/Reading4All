print("RUNNING UPDATED test.py")
import requests

API_URL = "https://hdzn5l02irp5ygnw.us-east-1.aws.endpoints.huggingface.cloud" # endpoint url 
HF_TOKEN = "<Insert Token Here>" # from hf -> settings -> access tokens 

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "image/png"   # or image/jpeg
}

image_path = "<Insert Test Image here>"

with open(image_path, "rb") as f:
    image_bytes = f.read()

print("Image bytes length:", len(image_bytes))

response = requests.post(
    API_URL,
    headers=headers,
    data=image_bytes
)

print(response.status_code)
print(response.json())

