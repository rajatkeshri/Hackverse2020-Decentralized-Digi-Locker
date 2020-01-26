import socket
import os
import cv2
import numpy as np

from PIL import Image

import os.path
import json

import hashlib
import face_recognition
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import pickle
import objcrypt
import pyAesCrypt
import time

import pyaudio
import wave
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib

import pandas as pd
import speech_recognition as sr
import re
from os import path
import re

import csv
import qrcode

import random
#######################################################################################
#-----------------------------------------------------------------
def samples(path):
        for i in range(0,5):
            #RECORD SPEAKERS VOICE FOR GIVING ATTENDANCE

            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 5
            WAVE_OUTPUT_FILENAME = path+"file"+str(i)+".wav"

            audio = pyaudio.PyAudio()

            # start Recording

            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            print(str(i)+" recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")

             # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

#---------------------------------------------------------------
def newtrain(speakers,name):
    #folder="C:/Anaconda codes/speaker reco/something new/for hack/add new people/"
    folder="C:/Anaconda codes/Hackverse/servermodel/clientfiles/"
    s=list(speakers)
    l=len(speakers)
    #name= input("enter your name")

    speakers.append(name)

    new_person=speakers[l]

    #rint(new_person)

    try:
        os.makedirs("clientfiles/dataset/"+ name)
    except:
        print("already exists")
        return(s)
    #os.mkdir(folder+"dataset/"+ name)


    x="clientfiles/dataset/"+name+"/"
    samples(x)

    training_speaker_name=name

    file_path=x
    file_names = os.listdir(file_path)
    #print((len(file_names)))


    lengths = np.empty(len(file_names))
    #print(np.shape(lengths))

    feature_vectors = np.empty([20,0])

    for i in range(len(file_names)):
        x, rate = librosa.load(file_path+file_names[i])               #loads the file
        #rate, x = wavfile.read(file_names[i])
        x=librosa.feature.mfcc(y=x[0:int(len(x)/1.25)], sr=rate)      #extracts mfcc

        #x = mfcc(x[0:len(x)/1.25], samplerate=rate)
        lengths[i] = int(len(x.transpose()))

        #print(np.shape(x))

        feature_vectors = np.concatenate((feature_vectors, x),axis=1)
        #feature_vectors = np.vstack((feature_vectors, x.transpose()))

    #print(((lengths)))
    #print(np.shape(feature_vectors))

    #TRAINING A MODEL


    N = 3  # Number of States of HMM
    Mixtures = 64# Number of Gaussian Mixtures.


    model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag')

    startprob = np.ones(N) * (10**(-30))  # Left to Right Model
    startprob[0] = 1.0 - (N-1)*(10**(-30))
    transmat = np.zeros([N, N])  # Initial Transmat for Left to Right Model
    #print(startprob,'\n',transmat)
    for i in range(N):
        for j in range(N):
            transmat[i, j] = 1/(N-i)
    transmat = np.triu(transmat, k=0)
    transmat[transmat == 0] = (10**(-30))


    model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag', init_params="mcw",n_iter=100)

    model.startprob_ = startprob
    model.transmat_ = transmat
    #print(startprob,'\n',transmat)

    feature=feature_vectors.transpose()
    #print(np.shape(feature))

    lengths = [ int(x) for x in lengths ]
    #print(type(lengths[0]))

    model.fit(feature,lengths)

    joblib.dump(model, folder+"/models/"+name+".pkl")
    return(speakers)


########################################################################################
def recvfile(client_socket,count,name,result):
    x = open('recieved/'+str(result)+'.json.aes','wb')
    while 1:
        data =''
        data = client_socket.recv(1024)
        if data == b'': break
        x.write(data)
    count+=1
    x.close()

def sendfile(filename,conn):
    x=open(filename,"rb")
    while 1:
        data = x.read(1024)
        if data == b'':
            print("finished")
            x.close()
            break
        conn.send(data)
    x.close()

def caputurecam(name,result):

    path="image_to_compare/"

    if not os.path.exists(path + name+ result):
        os.makedirs(path +result)

        cam = cv2.VideoCapture(0)

        cv2.namedWindow("test")
        img_counter = 0
        count=0

        while True:
            if count==4:
                break
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "image{}.png".format(img_counter)
                cv2.imwrite(path+result+"/"+img_name, frame)
                print("{} written!".format(img_name))

                img_counter += 1
                count+=1
        cam.release()

    else:
        print("user already exists")


def client_program():
    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    name=input("enter name")
    gender=input("enter gender")
    dob=input("enter dob/year of birth")
    aadhar=input("enter the aadhar number")

    date=datetime.now()

    data_entered={}
    data_entered["Name"]=name
    data_entered["Gender"]=gender
    data_entered["DOB"]=dob
    data_entered["AADHAR"]=aadhar
    data_entered["Timestamp"]=str(date)
    #data_entered["ImagePath"]=str("./"+image_path)
    datastring=""
    count=0
    while name.lower().strip() != 'bye':
        code="1"
        client_socket.send(code.encode())  # send message
        time.sleep(2)

        client_socket.send(name.encode())  # send message
        client_socket.send(gender.encode())
        client_socket.send(dob.encode())
        client_socket.send(aadhar.encode())
        #data = client_socket.recv(1024).decode()  # receive response
        result = client_socket.recv(1024).decode()  # receive response
        #print(result)
        ##############################################3

        recvfile(client_socket,count,name,result)
        print("Json Recieved in encrypted form")
        print("Hash of Json :"+ result)
        time.sleep(2)

        """
        x = open('recieved/jsonfile'+str(count)+'.json','wb')
        while 1:
            data =''
            data = client_socket.recv(1024)
            if data == b'': break
            x.write(data)
        count+=1
        x.close()
        """

        ##############################################
        #print('Received from server: ' + data)  # show in terminal

        path="image_to_compare_client/"
        print("Taking 4 pics for facial reco")
        print("\nHit space to capture images")
        caputurecam(name,result)

        #################################################
        time.sleep(2)

        speakers =[]
        print("Training New Voice")

        speakers1=newtrain(speakers,name)
        with open('clientfiles/s.txt', 'w') as f:
            for item in speakers1:
                f.write("%s\n" % item)

        print("Training New Voice done!")
        ################################################


        n=input("1 to Quit, 2 To add one more person")


        if n==str(1):
            sendval="done"
            time.sleep(2)
            #client_socket.send(sendval.encode())
            break





    print("New user registerd")
    client_socket.close()  # close the connection

##################################################################################################

def record(name1):
    #RECORD SPEAKERS VOICE FOR GIVING AUTHENTICATION
    name=name1[0]

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "C:/Anaconda codes/Hackverse/servermodel/clientfiles/samples/"+name+".wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("finished recording")

     # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

################################################################33

def facialrecognition(input_image,folder_to_compare):

    images = os.listdir(folder_to_compare)
    #print(images)

    image_to_be_matched = cv2.imread(input_image)
    if len(face_recognition.face_encodings(image_to_be_matched))!=0:
        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
    else:
        return("none")


    x=""
    # iterate over each image
    for i in images:
        temp=os.listdir(folder_to_compare+"/"+i)
        number=len(temp)
        count=0


        for j in temp:
            current_image = face_recognition.load_image_file(folder_to_compare+"/" + i+"/"+j)

            current_image_encoded = face_recognition.face_encodings(current_image)[0]

            result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
            if result[0] == True:
                count+=1

        if count==number:
            return(i)


    return("none")

def view_json():
    #FACIAL RECOGNITION TO ACCESS THE BLOCKS
    ########################################################

    print("Taking 1 picture for facial recognition")
    img_name="new pic.png"
    count=0
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server


    while True:
        if count==1:
            break
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            count+=1
    cam.release()

    print("\nPerforming facial recognition......")
    input_image=img_name
    folder_to_compare='image_to_compare/'

    nameofperson1=facialrecognition(input_image,folder_to_compare)
    print("\nFacial recognition done!")
    ################################################################################
    #speakers =["nishant","padma","rajat","shreekar","shruthi"]

    print("\nPerforming voice recognition......")
    with open('clientfiles/s.txt','r+') as f:
        speakers = f.read().splitlines()
        speakers=speakers[:len(speakers)]

        print((speakers))


    record(speakers)

    threshold = 100
    l=2
    uppercutoff=20000
    lowercutoff=8000

    #open the test data and find its probability
    #compare it with test probability and print predictions

    student="C:/Anaconda codes/Hackverse/servermodel/clientfiles/samples/"+speakers[0]+".wav" #SPEAKERS VOICE STORED

    file_path1=""
    #file_path1="C:/Anaconda codes/speaker reco/something new/for hack/other students/"
    test_speech1 = student
    speech1, rate = librosa.core.load(test_speech1)     #EXTRACT MFCC AND ADD IT OT FEATURE VECTOR
    feature_vectors12 = librosa.feature.mfcc(y=speech1, sr=rate)

    features1=feature_vectors12.transpose()

    #GET THE PREDICTION VALUES FOR EVERY MODEL CREATED FOR EACH SPEAKER
    x=[]

    path ="C:/Anaconda codes/Hackverse/servermodel/clientfiles/models/"
    names = os.listdir(path)
    #print(names)
    h=[]
    for i in range(0,len(names)):
        m1=joblib.load("C:/Anaconda codes/Hackverse/servermodel/clientfiles/models/"+str(names[i]) )
        p1 = m1.score(features1)
        p1=abs(p1)
        x.append(p1)
        #print(m1.predict(features1),'\n')

    print("Voice recognition done!")
    y=x.index(min(x))
    #print(x)
    #print(x.index(min(x))+1)
    if min(x)<uppercutoff and min(x)>lowercutoff:

        #print("Hi "+speakers[y]+".How are you?")
        p=speakers[y]
        flag="True"
    else:
        flag="False"
        #print("cant recognise. Speak again")

    if nameofperson1!="none"  or flag=="True":
        value="True"

        print("hashcode of block found is ",nameofperson1)

        count=0
        while True:

            code="2"
            client_socket.send(code.encode())
            print("-----------")
            time.sleep(2)

            print("Sending of hash code of block to server to get the decrypt key...")
            client_socket.send(value.encode())  # send message
            client_socket.send(nameofperson1.encode())

            decrypt_key = client_socket.recv(1024).decode()
            print("key to decrypt recieved from server",decrypt_key)

            # decrypt
            bufferSize = 64 * 1024
            pyAesCrypt.decryptFile("recieved/"+nameofperson1+".json.aes", "recieved/decrypted/"+nameofperson1+".json", decrypt_key, bufferSize)


            print("Decryption done")

            f=open("recieved/decrypted/"+nameofperson1+".json","r")
            di=json.load(f)
            print("Your stored data",di)

            qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
            qr.add_data(str(di))
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            img.save('qrcode_test.png')

            break
    else:
        while True:
            value="False"
            code="2"
            client_socket.send(code.encode())
            time.sleep(2)
            client_socket.send(value.encode())

        print("Do not try to steal data")

    client_socket.close()  # close the connection





if __name__ == '__main__':

    n=input("1 to add, 2 to View\n")

    if n==str(1):
        client_program()
    else:
        view_json()
