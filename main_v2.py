# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:33:39 2017

@author: AMD
"""
import vrep
import sys
import math
import numpy as np
from HmmClass import HmmScaled
import multiprocessing as thread
import pickle
import pdb
#pdb.set_trace()

#import matplotlib.pyplot as plt
#import copy
#import scipy
import time
from csvHandler import csvHandler
n = 6 #number of possible states
m = 181 #number of possible observations
num_gestures = 4 #number of possible gestures
num_models = 5 #number of models
hmm = []

for i in range(num_models):
    hmm.append([])
    for j in range(num_gestures):
        if (j == 0):
            with open ("estender_dedo"+str(i), 'rb') as fp:
                hmm[i].append(pickle.load(fp))
                #print(hmm[i][j].getName())
        elif(j == 1):
            with open ("flexionar_dedo"+str(i), 'rb') as fp:
                hmm[i].append(pickle.load(fp))
                #print(hmm[i][j].getName())
        elif(j == 2):
            with open ("dedo_esticado_estatico"+str(i), 'rb') as fp:
                hmm[i].append(pickle.load(fp))
                #print(hmm[i][j].getName())
        elif(j == 3):
            with open ("dedo_flexionado_estatico"+str(i), 'rb') as fp:
                hmm[i].append(pickle.load(fp))
                #print(hmm[i][j].getName())

#for i in range(num_models):
#    hmm.append([])
#    for j in range(num_gestures):
#        if(j == 0):
#            hmm[i].append(HmmScaled("estender_dedo"+str(i),n,m))
#        elif(j == 1):
#            hmm[i].append(HmmScaled("flexionar_dedo"+str(i),n,m))
#        elif(j == 2):
#            hmm[i].append(HmmScaled("dedo_esticado_estatico"+str(i),n,m))
#        elif(j == 3):
#            hmm[i].append(HmmScaled("dedo_flexionado_estatico"+str(i),n,m))

################################################################################
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

def getAngle(accel_X, accel_Y, accel_Z, accel_ref_X, accel_ref_Y, accel_ref_Z):
    'Calculates the angle between the acceleration vector and the reference vector '
    '''
    accel_x : acceleration array in the x axis
    accel_y : acceleration array in the y axis
    accel_z : acceleration array in the z axis
    returns a array of angles
    '''
    angle = []
    for i in range(len(accel_X)):
        norm_ref = np.sqrt(accel_ref_X[i]**2 + accel_ref_Y[i]**2 + accel_ref_Z[i]**2) #norm of the projection of the acceleration vector in the xy plane
        norm_accel = np.sqrt(accel_X[i]**2 + accel_Y[i]**2 + accel_Z[i]**2) #norm of the acceleration vector
        if(norm_ref > 0 and norm_accel > 0):
            aux = ( math.acos( (accel_X[i]*accel_ref_X[i] + accel_Y[i]*accel_ref_Y[i] + accel_Z[i]*accel_ref_Z[i])/(norm_ref*norm_accel) )) #finding the angle between the acceleration and the projection
            angle.append (int(180* aux/ math.pi)) #transform the angle from radian to degrees    
        else:
            angle.append(90) #if the projection norm is zero, then the angle is assumed to be 90 degrees
    return angle

#def getRefAngle(v_angle, v_angle_ref):
#    for i in range(len(v_angle)):
#        v_angle[i] = abs(v_angle[i] - v_angle_ref[i])
#    return v_angle
                       
def getAccelFromVrep(signal_name):
    e, a = vrep.simxGetStringSignal(clientID, signal_name, vrep.simx_opmode_oneshot_wait)
    a = vrep.simxUnpackFloats(a)
    vrep.simxClearStringSignal(clientID, signal_name, vrep.simx_opmode_oneshot)
    if e == vrep.simx_return_ok and a != None:
        return a
    else:
        raise Exception("acceleration not found!!!")


def getGesture(hmm, O):
    index = []
    prob = np.zeros((num_models,num_gestures))
    for i in range(0,num_models):
        for j in range(0,num_gestures):
            prob[i][j] = (hmm[i][j].computeProb(O[i], len(O[i]))) #using problem 1 to find log(P(O/model))
    print(prob)
    for i in range(num_models):
        index.append(np.argmax(prob[i])) # get the most probaly gesture for each finger
    print(index)
    #dynamical gestures
    if(index[0] == 1 and index[1] == 1 and index[2] == 1 and index[3] == 1 and index[4] == 1):
        print("Fechar a mao")
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Fechar a mao",vrep.simx_opmode_oneshot)
        return 1
    elif(index[0] == 1 and index[1] == 3 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        print("Fechar a mao")
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Fechar a mao",vrep.simx_opmode_oneshot)
        return 1
    elif(index[0] == 1 and index[1] == 1 and index[2] == 3 and index [3] == 3 and index[4] == 3):
        print("Fechar a mao")
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Fechar a mao",vrep.simx_opmode_oneshot)
        return 1
    elif(index[0] == 0 and index[1] == 0 and index[2] == 0 and index[3] == 0 and index[4] == 0):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Abrir a mao",vrep.simx_opmode_oneshot)
        print("Abrir a mao")
        return 0
    elif(index[0] == 2 and index[1] == 0 and index[2] == 0 and index[3] == 0 and index[4] == 0):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Abrir a mao",vrep.simx_opmode_oneshot)
        print("Abrir a mao")
        return 0
    elif(index[0] == 2 and index[1] == 2 and index[2] == 0 and index[3] == 0 and index[4] == 0):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Abrir a mao",vrep.simx_opmode_oneshot)
        print("Abrir a mao")
        return 0
    elif(index[0] == 0 and index[1] == 3 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Apontar",vrep.simx_opmode_oneshot)
        print("Apontar")
        return 2
    elif(index[0] == 2 and index[1] == 1 and index[2] == 1 and index[3] == 1 and index[4] == 1):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Apontar",vrep.simx_opmode_oneshot)
        print("Apontar")
        return 2
    elif(index[0] == 2 and index[1] == 1 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Apontar",vrep.simx_opmode_oneshot)
        print("Apontar")
        return 2
    elif(index[0] == 2 and index[1] == 2 and index[2] == 1 and index[3] == 1 and index[4] == 1):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Dois",vrep.simx_opmode_oneshot)
        print("Dois")
        return 3
    elif(index[0] == 0 and index[1] == 0 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Dois",vrep.simx_opmode_oneshot)
        print("Dois")
        return 3
    elif(index[0] == 2 and index[1] == 0 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Dois",vrep.simx_opmode_oneshot)
        print("Dois")
        return 3
    
    #static gestures
    elif all(i == 2 for i in index):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Abrir a mao",vrep.simx_opmode_oneshot)
        print("Abrir a mao")
        return 0
    elif all(i == 3 for i in index):
        print("Fechar a mao")
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Fechar a mao",vrep.simx_opmode_oneshot)
        return 1
    elif(index[0] == 2 and index[1] == 3 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Apontar",vrep.simx_opmode_oneshot)
        print("Apontar")
        return 2
    elif(index[0] == 2 and index[1] == 2 and index[2] == 3 and index[3] == 3 and index[4] == 3):
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Dois",vrep.simx_opmode_oneshot)
        print("Dois")
        return 3
    else:
        vrep.simxSetStringSignal(clientID,"recognized_gesture","Gesto nao reconhecido",vrep.simx_opmode_oneshot)
        return -1
    

if clientID != -1:
    print("connected to remote API server")
else :
    print("connection not succesful")
    sys.exit("could not connect")
#reading accelerations
##############################ref
error, accel_X_ref = vrep.simxGetFloatSignal(clientID, "accelerometerX",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y_ref = vrep.simxGetFloatSignal(clientID, "accelerometerY",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")

error, accel_Z_ref = vrep.simxGetFloatSignal(clientID, "accelerometerZ",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
###########################indicador    
error, accel_X1 = vrep.simxGetStringSignal(clientID, "accelerometerX1",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y1 = vrep.simxGetStringSignal(clientID, "accelerometerY1",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
    
error, accel_Z1 = vrep.simxGetStringSignal(clientID, "accelerometerZ1",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
###########################meio    
error, accel_X2 = vrep.simxGetStringSignal(clientID, "accelerometerX2",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y2 = vrep.simxGetStringSignal(clientID, "accelerometerY2",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
    
error, accel_Z2 = vrep.simxGetStringSignal(clientID, "accelerometerZ2",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")

###########################anelar    
error, accel_X3 = vrep.simxGetStringSignal(clientID, "accelerometerX3",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y3 = vrep.simxGetStringSignal(clientID, "accelerometerY3",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Z3 = vrep.simxGetStringSignal(clientID, "accelerometerZ3",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")

###########################mindinho  
error, accel_X4 = vrep.simxGetStringSignal(clientID, "accelerometerX4",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y4 = vrep.simxGetStringSignal(clientID, "accelerometerY4",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Z4 = vrep.simxGetStringSignal(clientID, "accelerometerZ4",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")

###########################polegar
error, accel_X5 = vrep.simxGetStringSignal(clientID, "accelerometerX5",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Y5 = vrep.simxGetStringSignal(clientID, "accelerometerY5",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")
    
error, accel_Z5 = vrep.simxGetStringSignal(clientID, "accelerometerZ5",vrep.simx_opmode_streaming)
if error == vrep.simx_return_ok:
    print("acceleration read with sucess")

########################
#get the correct gesture
gesture =[]
error, real_gesture = vrep.simxGetFloatSignal(clientID, "gesture", vrep.simx_opmode_streaming)
error, gesture.append(vrep.simxGetFloatSignal(clientID, "gesture1", vrep.simx_opmode_streaming))
error, gesture.append(vrep.simxGetFloatSignal(clientID, "gesture2", vrep.simx_opmode_streaming))
error, gesture.append(vrep.simxGetFloatSignal(clientID, "gesture3", vrep.simx_opmode_streaming))
error, gesture.append(vrep.simxGetFloatSignal(clientID, "gesture4", vrep.simx_opmode_streaming))
error, gesture.append(vrep.simxGetFloatSignal(clientID, "gesture5", vrep.simx_opmode_streaming))

e, canRead = vrep.simxGetIntegerSignal(clientID, "read", vrep.simx_opmode_streaming)
e, canRead = vrep.simxGetIntegerSignal(clientID, "read", vrep.simx_opmode_oneshot_wait)
v_angle = []

aux2 =1
init_time = time.time()
file = csvHandler("Acertos")
while aux2 == 1:
    e, canRead = vrep.simxGetIntegerSignal(clientID, "read", vrep.simx_opmode_oneshot_wait)
    if(canRead == 1):
        vrep.simxClearIntegerSignal(clientID, "read", vrep.simx_opmode_oneshot)
        try:


            vrep.simxSetIntegerSignal(clientID, "recognizing", 1, vrep.simx_opmode_oneshot)
            accel_X_ref = getAccelFromVrep("accelerometerX")
            accel_Y_ref = getAccelFromVrep("accelerometerY")
            accel_Z_ref = getAccelFromVrep("accelerometerZ")
            
            accel_X1 = getAccelFromVrep("accelerometerX1")
            accel_Y1 = getAccelFromVrep("accelerometerY1")  
            accel_Z1 = getAccelFromVrep("accelerometerZ1")
        
            accel_X2 = getAccelFromVrep("accelerometerX2")
            accel_Y2 = getAccelFromVrep("accelerometerY2")  
            accel_Z2 = getAccelFromVrep("accelerometerZ2")
            
            accel_X3 = getAccelFromVrep("accelerometerX3")
            accel_Y3 = getAccelFromVrep("accelerometerY3")  
            accel_Z3 = getAccelFromVrep("accelerometerZ3")
            
            accel_X4 = getAccelFromVrep("accelerometerX4")
            accel_Y4 = getAccelFromVrep("accelerometerY4")  
            accel_Z4 = getAccelFromVrep("accelerometerZ4")
            
            accel_X5 = getAccelFromVrep("accelerometerX5")
            accel_Y5 = getAccelFromVrep("accelerometerY5")  
            accel_Z5 = getAccelFromVrep("accelerometerZ5")
            error, real_gesture = vrep.simxGetFloatSignal(clientID, "gesture", vrep.simx_opmode_oneshot_wait)
            vrep.simxClearStringSignal(clientID, "gesture", vrep.simx_opmode_oneshot)
            error, gesture[0] = vrep.simxGetFloatSignal(clientID, "gesture1", vrep.simx_opmode_oneshot_wait)
            error, gesture[1] = vrep.simxGetFloatSignal(clientID, "gesture2", vrep.simx_opmode_oneshot_wait)
            error, gesture[2] = vrep.simxGetFloatSignal(clientID, "gesture3", vrep.simx_opmode_oneshot_wait)
            error, gesture[3] = vrep.simxGetFloatSignal(clientID, "gesture4", vrep.simx_opmode_oneshot_wait)
            error, gesture[4] = vrep.simxGetFloatSignal(clientID, "gesture5", vrep.simx_opmode_oneshot_wait)
            v_angle = []
            v_angle.append( getAngle(accel_X1,accel_Y1,accel_Z1, accel_X_ref, accel_Y_ref, accel_Z_ref))
            v_angle.append( getAngle(accel_X2,accel_Y2,accel_Z2, accel_X_ref, accel_Y_ref, accel_Z_ref))
            v_angle.append( getAngle(accel_X3,accel_Y3,accel_Z3, accel_X_ref, accel_Y_ref, accel_Z_ref))
            v_angle.append( getAngle(accel_X4,accel_Y4,accel_Z4, accel_X_ref, accel_Y_ref, accel_Z_ref))
            v_angle.append( getAngle(accel_X5,accel_Y5,accel_Z5, accel_X_ref, accel_Y_ref, accel_Z_ref))
            print(v_angle)
            time.sleep(2)
#            for i in range(0,num_models):
#                hmm[i][int(gesture[i])].A, hmm[i][int(gesture[i])].B, hmm[i][int(gesture[i])].pi =hmm[i][int(gesture[i])].train_scaled(len(v_angle[i]),v_angle[i])

            predicted = getGesture(hmm, v_angle)

            total_time = time.time() - init_time 
            if(predicted == real_gesture):
                file.save(str(int(real_gesture)) + ',' + str(predicted) + ',' + "1" + ',' + str(total_time), ',')
            else:
                file.save(str(int(real_gesture)) + ',' + str(predicted) + ',' + "0" + ',' + str(total_time), ',')
#            fft = scipy.fftpack.fft(v_angle[0])
            vrep.simxSetIntegerSignal(clientID, "recognizing", 0, vrep.simx_opmode_oneshot)
#            for i in range(5):
#                plt.figure(i)
#                plt.plot(v_angle[i])
            
        except Exception as error:
            aux1, aux2 = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
            print(error.args)
            continue

    


    aux1, aux2 = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
