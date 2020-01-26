import socket
import os
import cv2
import numpy as np

from PIL import Image

import os.path
import json
import os

import hashlib
import face_recognition
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import pickle
import glob
import objcrypt
import pyAesCrypt

import time
import random
import string
#####################################################################

def storehash(name,hashcode):
    data={}
    with open('hashcode.json', 'a+') as fp:
        try:
            data = json.load(fp)
            print(data)
        except:
            pass
        data[hashcode]=name
        print(data)
        json.dump(data, fp)


####################################################################
def newuser(data_entered,count,aadharno,server_socket,key):

    datastring=""
    name = data_entered["Name"]
    gender = data_entered["Gender"]
    dob = data_entered["DOB"]
    aadhar = data_entered["AADHAR"]
    #-----------------------------------------------------------------------
    flag1=0
    jsonpath="json_files"


    l=len(os.listdir(jsonpath))

    print("Creating block...")

    #CREATING THE BLOCKCHAIN
    if l==0:
        #IF THIS IS FIRST USER IN THE CHAIN
        aadharno.append(aadhar)
        f=open("aadharnumber.txt","w+")
        f.write(data_entered["AADHAR"]+"\n")
        f.close()

        f=open("temp.txt","w+")

        hashf=0
        hashf=str(hashf)
        data_entered["Hash"]=hashf

         #SHA256 ALGO
        for i in data_entered:
            datastring=datastring+data_entered[i]

        result = hashlib.sha256(datastring.encode())
        result=result.hexdigest()
        result=str(result)
        storehash(key,result)

        #CREATING BLOCK AND NAMNG IT AS HASH VALUE OF THE BLOCK
        fp=open(jsonpath+"/"+name+result+".json","w+")
        json.dump(data_entered, fp)


        f.write(result)
        f.close()
        datastring=""

        #############################################
        print("Block created")
        print("Initial hash",hashf)
        print("calculated hash for this block" +result)

        return(result)

    else:

        with open("aadharnumber.txt","r+") as f:
            content = f.readlines()
            print("-----------",content)
        content = [x.strip() for x in content]
        f.close()

        if data_entered["AADHAR"] in content:
            print("user with this aadhar exists")
            return("no",count)
        else:
            aadharno.append(aadhar)
            f=open("aadharnumber.txt","a+")
            f.write(data_entered["AADHAR"]+"\n")
            f.close()

            #READING PREVIOUS HASH VALUE
            f=open("temp.txt","r+")
            x=f.read()
            f.close()

            #SHA256 ALGO
            for i in data_entered:
                datastring=datastring+data_entered[i]
            #print(datastring)

            result = hashlib.sha256(datastring.encode())
            result=result.hexdigest()
            result=str(result)
            storehash(key,result)
            #print(result)

            #WRITING THE NEW BLOCK
            hashf=str(x)
            data_entered["Hash"]=hashf
            fp=open(jsonpath+"/"+name+result+".json","w+")
            json.dump(data_entered, fp)

            #WRITING HASH IN TEXT FILE
            f=open("temp.txt","w")
            f.write(result)
            f.close()

            #############################################
            """
            filetosend = open(jsonpath+"/"+name+result+".json", "r")


            data = filetosend.read(1024).encode()

            while data:
                print("Sending...")
                server_socket.send(data)
                data = filetosend.read(1024).encode()
                server_socket.send("DONE".encode())
            filetosend.close()
            """
            #############################################

            print("Block created")
            print("Previous",hashf)
            print("Calculated hash for this block" +result)

            return(result)

#####################################################################
def randomString(stringLength=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
###########################################################################
def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 5000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(5)
    #conn, address = server_socket.accept()  # accept new connection
    #print("Connection from: " + str(address))
    #print("Enter 'quit' to exit")
    #message = input(" -> ")

    while True:
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))


        code = conn.recv(1024).decode()
        if code ==str(1):

            print("Entering New Record creation")
            # receive data stream. it won't accept data packet greater than 1024 bytes
            jsonpath="json_files"
            name = conn.recv(1024).decode()
            gender = conn.recv(1024).decode()
            dob = conn.recv(1024).decode()
            aadhar = conn.recv(1024).decode()

            print("from connected user:name, gender, dob, aadhar\n ")
            print(name,gender,dob,aadhar)

            date=datetime.now()

            data_entered={}
            data_entered["Name"]=name
            data_entered["Gender"]=gender
            data_entered["DOB"]=dob
            data_entered["AADHAR"]=aadhar
            data_entered["Timestamp"]=str(date)
            #data_entered["ImagePath"]=str("./"+image_path)
            datastring=""

            ########################################################

            password = randomString()
            key=password
            crypter = objcrypt.Crypter('key', password)
            ########################################################
            count=0
            name_d={}
            aadharno=[]
            result=newuser(data_entered,count,aadharno,server_socket,key)

            conn.send(result.encode())
            #########################################################


            print("Doing encryption of json")
            print("Encryption key generated:"+ str(password))
            bufferSize = 64 * 1024
            # encrypt
            pyAesCrypt.encryptFile(jsonpath+"/"+name+result+".json", jsonpath+"/"+name+result+".json.aes", password, bufferSize)
            # decrypt
            #pyAesCrypt.decryptFile("data.txt.aes", "dataout.txt", password, bufferSize)jsonpath+"/"+name+result+".json.aes"
            print("Done encryption!")

            x=open(jsonpath+"/"+name+result+".json.aes","rb")
            while 1:
                data = x.read(1024)
                if data == b'':
                    #print("finished sending")
                    x.close()
                    break
                conn.send(data)
            x.close()


            print("Json sent")

            #exitval = conn.recv(1024).decode()
            #if exitval=="done":
            #########################################################
            #time.sleep(10)
            #break
        #files = glob.glob(jsonpath+"/")
        #for f in files:
        #    os.remove(f)

        else:
            print("Entering view mode")
            value = conn.recv(1024).decode()
            #print(value)
            hashvalue = conn.recv(1024).decode()
            #print(hashvalue)
            print("Hash of block from clien:",hashvalue)

            if value=="True":
                f=open("hashcode.json","r")
                di=json.load(f)

                decrypt_key=di[hashvalue]
                print("Sending the decrypted key:",decrypt_key)

                conn.send(decrypt_key.encode())
            else:
                print("Theft")

            n=input(" Press 1 to Turn off, 0 to continue")

            if n==str(1):
                break


        conn.close()




if __name__ == '__main__':
    server_program()
