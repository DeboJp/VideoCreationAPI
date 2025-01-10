import requests
 
url = "https://api.goapi.ai/api/v1/task/ad735171-7089-4fbe-bd59-9a08cabf5efb" 
 
payload={}
headers = {
   'X-API-KEY': '',
  'Content-Type': 'application/json'
}
 
response = requests.request("GET", url, headers=headers, data=payload)
 
print(response.text)
