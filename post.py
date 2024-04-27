import requests
import json

url = 'http://127.0.0.1:8000/prompt'
data = {'query': 'I want a car that is reliable. Which car should I purchase?'}

response = requests.post(url, json=data)

if response.status_code == 200:
    print('Request successful!')
    print('Response:', response.json())
else:
    print('Request failed with status code:', response.status_code)
    print('Response:', response.text)
