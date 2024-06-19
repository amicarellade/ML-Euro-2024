import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

response = requests.get('https://www.sofascore.com/football/rankings/fifa')
print(response.status_code)

soup = BeautifulSoup(response.text, 'html.parser')

cookies = {
    '_gcl_au': '1.1.1898391120.1718334564',
    '_ga': 'GA1.1.1029693527.1718334564',
    '_li_dcdm_c': '.sofascore.com',
    '_lc2_fpi': 'a78faec1e09d--01j0abjkr03emmhmxbsarw3fvc',
    '_lc2_fpi_meta': '{%22w%22:1718334607104}',
    '__gads': 'ID=9f758228b593eeee:T=1718334568:RT=1718335820:S=ALNI_MbCnYXct7bCKgtkeZDz7MZ1gLWYHQ',
    '__gpi': 'UID=00000e2abdd0a74e:T=1718334568:RT=1718335820:S=ALNI_MY2WkAqEsYB4aPP5P1Z4TfSkvac0w',
    '__eoi': 'ID=b4265eab2b624d07:T=1718334568:RT=1718335820:S=AA-Afjbh7SaNsUWxCHXU9ksePLXx',
    'FCNEC': '%5B%5B%22AKsRol9UNVrjqhWZ9RgjLM6WwjzLPdPWcVT6ja015wAtCdVNLClKJDLe6msEZ4fJu8ijsQaVxJxe78-_K0l_EBjG8xJs5VNPIRcw-X2EGU1x9r_33aOLMGT1HT71PMLl-xh63DOFChGA6-SaZJZJRCP54IdgripNGA%3D%3D%22%5D%5D',
    '_ga_3KF4XTPHC4': 'GS1.1.1718334564.1.1.1718335968.58.0.0',
    '_ga_QH2YGS7BB4': 'GS1.1.1718334564.1.1.1718335968.0.0.0',
    '_ga_HNQ9P9MGZR': 'GS1.1.1718334564.1.1.1718335968.58.0.0',
}

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'no-cache',
    # 'cookie': '_gcl_au=1.1.1898391120.1718334564; _ga=GA1.1.1029693527.1718334564; _li_dcdm_c=.sofascore.com; _lc2_fpi=a78faec1e09d--01j0abjkr03emmhmxbsarw3fvc; _lc2_fpi_meta={%22w%22:1718334607104}; __gads=ID=9f758228b593eeee:T=1718334568:RT=1718335820:S=ALNI_MbCnYXct7bCKgtkeZDz7MZ1gLWYHQ; __gpi=UID=00000e2abdd0a74e:T=1718334568:RT=1718335820:S=ALNI_MY2WkAqEsYB4aPP5P1Z4TfSkvac0w; __eoi=ID=b4265eab2b624d07:T=1718334568:RT=1718335820:S=AA-Afjbh7SaNsUWxCHXU9ksePLXx; FCNEC=%5B%5B%22AKsRol9UNVrjqhWZ9RgjLM6WwjzLPdPWcVT6ja015wAtCdVNLClKJDLe6msEZ4fJu8ijsQaVxJxe78-_K0l_EBjG8xJs5VNPIRcw-X2EGU1x9r_33aOLMGT1HT71PMLl-xh63DOFChGA6-SaZJZJRCP54IdgripNGA%3D%3D%22%5D%5D; _ga_3KF4XTPHC4=GS1.1.1718334564.1.1.1718335968.58.0.0; _ga_QH2YGS7BB4=GS1.1.1718334564.1.1.1718335968.0.0.0; _ga_HNQ9P9MGZR=GS1.1.1718334564.1.1.1718335968.58.0.0',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://www.sofascore.com/football/rankings/fifa',
    'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36',
    'x-requested-with': 'dcc309',
}

response = requests.get('https://www.sofascore.com/api/v1/rankings/type/2', cookies=cookies, headers=headers)
rankings = response.json()

# data = json.loads(rankings)

# Extract relevant information
rankings = rankings['rankings']
ranking_list = []

for item in rankings:
    team_name = item['team']['name']
    ranking = item['ranking']
    points = item['points']
    ranking_list.append({'team': team_name, 'ranking': ranking, 'points': points})

# Convert to DataFrame
fifa_rankings_df = pd.DataFrame(ranking_list)

# Inspect the DataFrame
print(fifa_rankings_df)
# fifa_rankings_df.to_csv('fifa_rankings.csv', index=False)