# coding=utf-8
from __future__ import print_function
import time
import requests
import cv2
import operator
import codecs
import numpy as np
import os
import time

# Import library to display results
import matplotlib.pyplot as plt

_url_recognize = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
_url_recognizeinvideo = 'https://api.projectoxford.ai/emotion/v1.0/recognizeinvideo'
_url_trackface = 'https://api.projectoxford.ai/video/v1.0/trackface'
_url_identify = 'https://api.projectoxford.ai/face/v1.0/identify'
_url_persongroup = 'https://api.projectoxford.ai/face/v1.0/persongroups/hermione'
_url_detect = 'https://api.projectoxford.ai/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'

_key_emotion = '57581a86d1854feeb598110c69eccb8a'  # primary key - emotion
_key_video = '75e7945f0f4c460fb4039d967b4210af'  #  primary key - video
_key_face = '8b457a38902d463f9a21af2620514c3a'   #  primary key - face

_maxNumRetries = 10


def processRequest(method, url, json, data, headers, params):
    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        if method == 'patch':
            response = requests.patch(url, files=data, headers=headers)
        else:
            response = requests.request(method, url, json=json, data=data, headers=headers, params=params)

        if response.status_code == 429:

            print("Message: %s" % (response.json()['error']['message']))

            if retries <= _maxNumRetries:
                time.sleep(1)
                retries += 1
                continue
            else:
                print('Error: failed after retrying!')
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
                result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower():
                    result = response.content

        elif response.status_code == 202:

            _getVideoData = response.headers['Operation-Location']
            if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
                result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower():
                    result = response.content
            print(_getVideoData)
            print(response.headers)
            print(response)

        else:
            print("Error code: %d" % (response.status_code))
            print("Message: %s" % (response.json()['error']['message']))

        break

    return result


def createPersonGroup():
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json'
    params = None
    json = {"name": "hermione"}
    data = None
    result = processRequest('put', _url_persongroup, json, data, headers, params)
    print("Group created")


def createPerson():
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json'
    params = None
    json = {"name": "hermione"}
    data = None

    result = processRequest('post', _url_persongroup + '/persons', json, data, headers, params)
    print(result)


def addFace(personID):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/octet-stream'
    params = None
    json = None
    data = None

    s = r'images/face_examples/%d.jpg'
    for i in [1, 2, 3, 4, 5]:
        with codecs.open(s % i, 'rb') as f:
            result = processRequest('post', _url_persongroup + '/persons/' + personID + '/persistedFaces', json,
                                    f.read(), headers,
                                    params)
            print(result)
            print("Added image")


def trainPersonGroup():
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json'
    params = None
    json = {"name": "hermione"}
    data = None

    result = processRequest('post', _url_persongroup + '/train', json, data, headers, params)
    print(result)


def getTrainStatus():
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json'
    params = None
    json = None
    data = None

    result = processRequest('get', _url_persongroup + '/training', json, data, headers, params)
    print(result)


def identify(arr):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json'
    params = None
    json = {
        "personGroupId": "hermione",
        "faceIds": arr,
        "maxNumOfCandidatesReturned": 1,
        "confidenceThreshold": 0.5
    }
    data = None

    result = processRequest('post', _url_identify, json, data, headers, params)
    return result


def findPerson(img):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/octet-stream'
    params = None
    json = None
    with open(img, 'rb') as f:
        data = f.read()

    result = processRequest('post', _url_detect, json, data, headers, params)
    return result


def getEmotions(img):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_emotion
    headers['Content-Type'] = 'application/octet-stream'
    params = None
    json = None
    with open(img, 'rb') as f:
        data = f.read()
    result = processRequest('post', _url_recognize, json, data, headers, params)

    return result


# createPersonGroup() # создали группу персонажей
# a = createPerson()['personId']  # создали главного персонажа
# addFace(a) # добавили лица персонажа
# trainPersonGroup() # тренируем узнавать персонажа
#
# getTrainStatus() # проверяем, завершилась ли тренировка

results = codecs.open('results.csv', 'w', 'utf-8')
results.write('\t'.join(['picnum', 'sadness', 'neutral', 'contempt', 'disgust', 'anger', 'surprise', 'fear', 'happiness']) + '\r\n')
for root, dirs, files in os.walk('images/movie_shots'):
    for f in files:
        try:
            time.sleep(20)
            img = os.path.join(root, f)
            people = findPerson(img)
            main_hero = identify([d['faceId'] for d in people])
            faceID = ''
            faceRectangle = {}
            print(main_hero)

            for i in main_hero:
                if i['candidates']:
                    faceID = i['faceId']
                    for p in people:
                        if p['faceId'] == faceID:
                            faceRectangle = p['faceRectangle']
                            break
                    break

            if faceID and faceRectangle:
                emotions = getEmotions(img)
                for p in emotions:
                    if p['faceRectangle'] == faceRectangle:
                        r = '\t'.join(str(i) for i in [f, p['scores']['sadness'],
                                                       p['scores']['neutral'],
                                                       p['scores']['contempt'],
                                                       p['scores']['disgust'],
                                                       p['scores']['anger'],
                                                       p['scores']['surprise'],
                                                       p['scores']['fear'],
                                                       p['scores']['happiness']])
                        results.write(r + '\r\n')
                        print(r)
        except:
            pass

results.close()