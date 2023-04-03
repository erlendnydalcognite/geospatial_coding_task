import requests
import json

with open('shapes.json', 'r') as f:
    data = json.load(f)

# Convert the data to a JSON string
json_data = json.dumps(data)

# Set the headers for the request
headers = {'Content-Type': 'application/json'}

app_url = "https://spacemaker-qpomukupha-ew.a.run.app/main"
response = requests.post(app_url, data=json_data, headers=headers)
print(response.text)