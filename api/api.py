import requests
from NLPDataCollection.constants import YOUTUBE_API_KEY

api_key = YOUTUBE_API_KEY

url = f"https: // youtube.googleapis.com/youtube/v3/channels?part = snippet % 2CcontentDetails % 2Cstatistics & id = UC_x5XG1OV2P6uZZ5FSM9Ttw & key = {
    api_key}"
headers = {
    "header": 'Authorization: Bearer',
    "header": 'Accept: application/json',
}

r = requests.get(url=url, headers=headers)
