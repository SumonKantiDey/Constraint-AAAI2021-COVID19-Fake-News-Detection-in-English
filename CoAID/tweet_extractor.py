"""
Fetch a single tweet as JSON using its id.
"""
from __future__ import print_function
import pandas as pd
import os
import json
import twarc
import time
import argparse
path = "/content/drive/MyDrive/COVID19 Fake News Detection in English/CoAID/05-01-2020/"
def _store_data(data):
    if os.path.exists(f"{path}05-01-2020real_tweet.json") == False:
        store_data = []
        store_data.append(data)
        with open(f'{path}05-01-2020real_tweet.json', 'w') as fp:
            json.dump(store_data, fp)
    else:
        with open(f'{path}05-01-2020real_tweet.json', 'r') as fp:
            load_data = json.load(fp)
        load_data.append(data)
        with open(f'{path}05-01-2020real_tweet.json', 'w') as fp:
            json.dump(load_data, fp)
    return


e = os.environ.get
parser = argparse.ArgumentParser("tweet.py")

parser.add_argument('--tweet_id', action="store", help="Tweet ID")
parser.add_argument("--consumer_key", action="store",
                    default='',
                    help="Twitter API consumer key")
parser.add_argument("--consumer_secret", action="store",
                    default='',
                    help="Twitter API consumer secret")
parser.add_argument("--access_token", action="store",
                    default='',
                    help="Twitter API access key")
parser.add_argument("--access_token_secret", action="store",
                    default='',
                    help="Twitter API access token secret")
args = parser.parse_args()

tw = twarc.Twarc(args.consumer_key, args.consumer_secret, args.access_token, args.access_token_secret)

if __name__ == "__main__":
    tweet_real_ids = pd.read_csv("/content/drive/MyDrive/COVID19 Fake News Detection in English/CoAID/05-01-2020/NewsRealCOVID-19_tweets.csv")
    cnt = 0
    for tweet_id in tweet_real_ids['tweet_id']:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
            time.sleep(20)
        try:
            tweet = tw.get('https://api.twitter.com/1.1/statuses/show/%s.json' % tweet_id)
            #print(json.dumps(tweet.json(), indent=2))
            real_tweets = {}
            real_tweets["id"] = tweet_id
            real_tweets["text"] = tweet.json()['full_text']
            _store_data(real_tweets)
        except Exception as error:
            #print(tweet_id)
            pass