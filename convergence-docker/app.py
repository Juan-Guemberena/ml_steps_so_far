import pickle
import json
import gensim
import numpy
import scipy


BUCKET_NAME = 'ml-model-convergence'
MODEL_FILE_NAME = 'glove.6B.100d_model.sav'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME


def handler(event, context):
  print(event)
  positive = []
  negative = []
  if 'body' in event:
    event = json.loads(event['body'])

  for i in event["positive"]:
    positive.append(i)
  for i in event["negative"]:
    negative.append(i)
  print(positive)
  print(negative)  
  try:
    response = load_model().most_similar(positive=positive,negative=negative, topn=5)
  except KeyError as e:
    return e
  return answer(response)

def load_model():
    return pickle.load(open(MODEL_FILE_NAME, 'rb'))

def answer (ans):
  response_object = {}
  response_object['statusCode'] = 200
  response_object['headers'] = {}
  response_object['headers']['Content-Type'] = "application/json"
  response_object['body'] = json.dumps(ans)
  return response_object
