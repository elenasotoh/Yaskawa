#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:37:31 2023

@author: elenasotoh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import seaborn as sns
from scipy import stats
from scipy.stats import iqr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import deque
import os
import random
import math

path = '/Users/elenasotoh/Desktop/FALL_2022/YASKAWA/ML/Experiments_Amplifier_2000'
dir_list = os.listdir(path)
#count # of trials
Number_of_Trials = 0 #total trials
Number_of_motions = 0

Number_of_Baseline_Trials = 4
folder_unfiltered = "fname_ai_sample_rate_1000Hz_sample_filtered"
motion = "HeadFlexion_m.csv"
Trial_name  = "Trial1"

for folder in dir_list:
    if folder_unfiltered in folder and Trial_name in folder and "_m" in folder:
        Number_of_motions +=1
    if folder_unfiltered in folder and motion in folder:
        Number_of_Trials +=1



dataframe_dictionary_baseline = {}
dataframe_dictionary_test = {}
motion_list = []
super_dataframe = []
# motion_dictionary = {'LateralFLexionRight':[], 'LateralFLexionLeft':[], 'HeadExtension':[], 'HeadFlexion':[], 'HeadRotationRight':[], 'HeadRotationLeft':[]}
motion_dictionary = {'HeadExtension':[], 'HeadFlexion':[], 'HeadRotationRight':[], 'HeadRotationLeft':[]}

for motion_in_dic in motion_dictionary.keys():
    motion_list.append(motion_in_dic )
    dataframe_dictionary_baseline[motion_in_dic] = []

trials_range = range(3,5) 

trial_m = 0

#read # of columns in the dataframe
df = pd.read_csv('/Users/elenasotoh/Desktop/FALL_2022/YASKAWA/ML/SecondSetofExperiments/fname_ai_sample_rate_1000Hz_sample_filtered_Trial1_HeadFlexion_m.csv')
total_cols=len(df.axes[1])

#create super dataframe with trials for baseline
for folder in dir_list:
    for motion in motion_list:
        if "_m" in folder and folder_unfiltered in folder and motion in folder and ".fname" not in folder:
            for trial in trials_range:
                if "Trial"+str(trial) in folder:
                    trial_m +=1
                    read_csv = pd.read_csv(path + '/' + folder_unfiltered + "_Trial" +str(trial)+"_"+motion+"_m.csv")
                    read_csv.insert(total_cols,"Motion Name", motion )
                    read_csv.insert(total_cols+1,"Trial Number", str(trial))
                    read_csv.insert(total_cols+2,"Contraction longer than 2 seconds?", "Y")
                    super_dataframe.append(read_csv)

test_dataframe = []
Number_of_Test_Trials = 4 
# test_trials_range = range(Number_of_Baseline_Trials+1, Number_of_Test_Trials+Number_of_Baseline_Trials+1) #change when more data available 
test_trials_range = range(5, 12) 
trial_m_test = 1
#create super dataframe with trials for baseline
for folder in dir_list:
    for motion in motion_list:
        if "_m" in folder and folder_unfiltered in folder and motion in folder and ".fname" not in folder:
            for trial in test_trials_range:
                if "Trial"+str(trial) in folder:
                    trial_m_test +=1
                    read_csv = pd.read_csv(path + '/' + folder_unfiltered + "_Trial" +str(trial)+"_"+motion+"_m.csv")
                    # print(path + '/' + folder_unfiltered + "_Trial" +str(trial)+"_"+motion+"_m.csv")
                    read_csv.insert(total_cols,"Motion Name", motion )
                    read_csv.insert(total_cols+1,"Trial Number", str(trial))
                    read_csv.insert(total_cols+2,"Contraction longer than 2 seconds?", "Y")
                    test_dataframe.append(read_csv)
    
def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

def test(test_dataframe, frameSize):
    start_time = 4.000
    add = 0
    # channel_list = ['EMG1', 'EMG2', 'EMG3','EMG4','EMG5','EMG6']  #editado
    channel_list = ['EMG1', 'EMG2','EMG3','EMG4','EMG5','EMG6']  #editado
    first_channel = channel_list[0]
    super_dataframe = test_dataframe
    samplingRate = 1000.0
    nyq = 0.5*samplingRate
    lowcut = 5/nyq   #7 #5 on right side, 10 on left lo tenia en 5
    highcut = [40/nyq, 400/nyq] #300 for left, 400 for right
    
    from scipy.signal import butter, lfilter, iirfilter
    b_low, a_low = butter(4, lowcut, btype='lowpass')
    b_high, a_high= butter(4, highcut, btype='bandpass')
    
    frameSize = frameSize #ms    #try 250
    maxlen_filter = int(frameSize/2)
    
    all_dataframes = {}
    all_dataframe_power = {}
    dataframe_count = 0
    counter_test_df = 0
    counter_test_df_add = 0
    for dataf in super_dataframe:

        time = dataf.iloc[:,0]
        time_list = list(time)

        stop = 0
        for t in time:
            if t >= start_time and stop == 0:
                startIndex = time_list.index(t)
                baseline_startindex = time_list.index(t)
                stop = 1

        # baseline_startindex = 0 # borrar dsps 

        dataframe_count += 1

        all_dataframes["dataframe{0}".format((dataframe_count))]= {}
        # for i in range(0,(int(cycles_normal))):
        #     motion_name =  dataf.iloc[1,17]
        #     counter_test_df += 1
        #     # all_dataframe_power["dataframe{0}".format((counter_test_df))]= {'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [motion_name]}  #editado
        #     all_dataframe_power["dataframe{0}".format((counter_test_df))]= { 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [motion_name]} 
            
        #     print("cycles_normal for dataframe", counter_test_df,":", cycles_normal)
            # all_dataframe_power["Motion Name"] = motion_name
    counter_df = 0
    for dataframe in super_dataframe:
        
        counter_df += 1
        motion_name =  dataframe.iloc[1,17]
        trial_number =  dataframe.iloc[1,18]
        # filtering_data = {'Time': [], 'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[]}
        # filtered_data = {'Time': [],'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [], "Trial": []}
        # power_data = {'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": []}
        filtering_data = {'Time': [], 'EMG1':[],'EMG2':[],'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[]}  #editado
        filtered_data = {'Time': [],'EMG1':[],'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [], "Trial": []}    #editado
        power_data = {'EMG1':[],'EMG2':[],'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": []}
        
        
        filtered_data["Motion Name"]= motion_name
        filtered_data["Trial"]= trial_number
        time = dataframe.iloc[:,0]
        time_list_dataframe = list(time)
        stop = 0

        for t in time:
            if t >= start_time and stop == 0:
                startIndex = time_list_dataframe.index(t)
                baseline_startindex = time_list_dataframe.index(t)
                stop = 1

        # startIndex = 0 # borrar dsps 
        # baseline_startindex = 0 # borrar dsps 
        # #filtering
        # cycles_filter = math.floor((len(dataframe[baseline_startindex:(len(dataframe))]))/maxlen_filter)
        # 
        cycles_normal = math.floor((len(dataframe[baseline_startindex:(len(dataframe))]))/frameSize)*2

        
        filtered_data['Time'] = dataframe.iloc[:,0].tolist()
        
        
        filtered_data['EMG1'] = dataframe.iloc[:,1].tolist()   #editado
        filtered_data['EMG1'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG1'])))))

        filtered_data['EMG2'] = dataframe.iloc[:,3].tolist()   #editado
        filtered_data['EMG2'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG2'])))))

        filtered_data['EMG3'] = dataframe.iloc[:,5].tolist()
        filtered_data['EMG3'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG3'])))))

        filtered_data['EMG4'] = dataframe.iloc[:,7].tolist()
        filtered_data['EMG4'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG4'])))))

        filtered_data['EMG5'] = dataframe.iloc[:,9].tolist()
        filtered_data['EMG5'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG5'])))))

        filtered_data['EMG6'] = dataframe.iloc[:,11].tolist()
        filtered_data['EMG6'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG6'])))))
            #Normalization
        for x in range(0,(len(filtered_data)-3)):
            
            # counter_test_df_add = 0
            channel = channel_list[x]
            # print(counter_df)
            # print('counter_df:', counter_df)
            # print(x)

            #time = dataframe.iloc[:,0] channel time was already defined above
            if "EMG" in channel:
                # print(channel)
                # counter_test_df_add += 1
                startIndex = baseline_startindex
                # real_time_normalized_overlap = []
                filtered_channel_data = filtered_data[channel]
                filtered_channel_data_squared = []
                list_to_sum = []
                
                if counter_df != 1: 
                    counter_test_df_add = (counter_df-1)*int(cycles_normal)
                else: 
                    counter_test_df_add = 0
                for i in range(0, (len(filtered_channel_data))):
                    i_squared = ((filtered_channel_data[i])*(filtered_channel_data[i]))
                    list_to_sum.append(i_squared)
                
                for i in range(0,int(cycles_normal)):
                    
                    # print(counter_test_df_add)
                    # all_dataframe_power["Motion Name"].append(motion_name)
                    stopIndex = int(startIndex + (frameSize))
                    timesegment = time_list[startIndex:stopIndex]
                    t1 = timesegment[0] #time_list_dataframe[startIndex]
                    t2 = timesegment[-1] #time_list_dataframe[stopIndex-1]  
                    # t1_list.append((t2-t1))
                    # t2_list.append(t2)
                    segment = list_to_sum[startIndex:(stopIndex)]  
                    segment_sum = sum(segment)
                    p =((segment_sum)/(t2-t1))
                    startIndex = int(stopIndex-(frameSize*.5))
                    power_data[channel].append(p)
                    counter_test_df_add += 1
                    # experimento.append(counter_test_df_add)  #borrar'
                    if channel== first_channel:
                        all_dataframe_power["dataframe{0}".format((counter_test_df_add))]= { 'EMG1':[],'EMG2':[],'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [motion_name]} 
                    all_dataframe_power["dataframe{0}".format(counter_test_df_add)][channel].append(p)
                    
        all_dataframes["dataframe{0}".format(counter_df)] = filtered_data
        
    return all_dataframe_power, cycles_normal, all_dataframes, stopIndex, time_list, list_to_sum, segment, cycles_normal, stopIndex

def baseline(super_dataframe, frameSize):

    super_dataframe = super_dataframe
    samplingRate = 1000.0
    start_time = 2.500
    nyq = 0.5*samplingRate
    lowcut = 30/nyq   #7 #5 on right side, 10 on left
    highcut = [40/nyq, 400/nyq] #300 for left, 400 for right
    
    from scipy.signal import butter, lfilter, iirfilter
    b_low, a_low = butter(4, lowcut, btype='lowpass')
    b_high, a_high= butter(4, highcut, btype='bandpass')
    
    frameSize = frameSize #ms    #try 250
    maxlen_filter = int(frameSize/2)
    
    all_dataframes = {}
    all_dataframe_power = {}
    dataframe_count = 0
    for dataframe in super_dataframe:
        dataframe_count += 1
        all_dataframes["dataframe{0}".format((dataframe_count))]= {}
        all_dataframe_power["dataframe{0}".format((dataframe_count))]= {}

    counter_df = 0
    for dataframe in super_dataframe:

        counter_df += 1
        motion_name =  dataframe.iloc[1,17]
        # filtering_data = {'Time': [], 'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[]}
        # filtered_data = {'Time': [],'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": []}
        # power_data = {'EMG1':[], 'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": []}

        filtering_data = {'Time': [], 'EMG1':[],'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[]}
        filtered_data = {'Time': [], 'EMG1':[],'EMG2':[], 'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": [], "Trial": []}
        power_data = {'EMG1':[],'EMG2':[],'EMG3':[], 'EMG4':[], 'EMG5':[], 'EMG6':[], "Motion Name": []}
        
        
        filtered_data["Motion Name"]= motion_name
        power_data["Motion Name"]= motion_name
        time = dataframe.iloc[:,0]
        time_list = list(time)
        stop = 0

        for t in time:
            if t >= start_time and stop == 0:
                startIndex = time_list.index(t)
                baseline_startindex = time_list.index(t)
                stop = 1
        #filtering
        
        cycles_normal = math.floor((len(dataframe[baseline_startindex:(len(dataframe))]))/frameSize)*2
        
        

        stopIndex = int((len(dataframe)))
                        
        filtered_data['Time'] = dataframe.iloc[startIndex:(stopIndex),0].tolist()
        
        filtered_data['EMG1'] = dataframe.iloc[startIndex:(stopIndex),1].tolist()
        filtered_data['EMG1'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG1'])))))

        filtered_data['EMG2'] = dataframe.iloc[startIndex:(stopIndex),3].tolist()
        filtered_data['EMG2'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG2'])))))

        filtered_data['EMG3'] = dataframe.iloc[startIndex:(stopIndex),5].tolist()
        filtered_data['EMG3'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG3'])))))

        filtered_data['EMG4'] = dataframe.iloc[startIndex:(stopIndex),7].tolist()
        filtered_data['EMG4'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG4'])))))

        filtered_data['EMG5'] = dataframe.iloc[startIndex:(stopIndex),9].tolist()
        filtered_data['EMG5'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG5'])))))

        filtered_data['EMG6'] = dataframe.iloc[startIndex:(stopIndex),11].tolist()
        filtered_data['EMG6'] = (lfilter(b_low, a_low,(abs(lfilter(b_high, a_high, filtered_data['EMG6'])))))


            
            #Normalization
        # startIndex = baseline_startindex
        for channel in filtered_data:
            #time = dataframe.iloc[:,0] channel time was already defined above
            list_to_sum = []
            t1 = time_list[baseline_startindex]
            t2 = time_list[-1]
            if "EMG" in channel:
                filtered_channel_data = filtered_data[channel]
                for i in range(0, (len(filtered_channel_data))):
                      i_squared = ((filtered_channel_data[i])*(filtered_channel_data[i]))
                      list_to_sum.append(i_squared)
                  # sum of all voltages squared to get the power
                p = sum(list_to_sum)/(t2-t1)
                power_data[channel].append(p)
            
        
        all_dataframe_power["dataframe{0}".format(counter_df)] = power_data
        all_dataframes["dataframe{0}".format(counter_df)] = filtered_data
            # except:
            #     continue

       
    return all_dataframe_power

frameSize = 300 #276
baseline = baseline(super_dataframe, frameSize)
test = test(test_dataframe, frameSize)

df1_normal = []
for item in test[0]:
    df_counter = item.split('frame')
    df_counter = int(df_counter[1])
    if df_counter <= test[1]:
        df1_normal.append(test[0][item]['EMG6'])
time_offline = test[2]['dataframe1']['Time']

def cross_correlation(baseline_frame, test_frames):
    
    baseline_frame = baseline_frame
    test_frames = test_frames
    
    # channel_counter = 0
    correction_values = 2
    correction_list = deque(maxlen=correction_values)
    
    motion_list_Not_repeated = []
    predicted_interval_label = []
    predicted_index_ca = []
    true_interval_label = []
    test_list = []
    baseline_list = []   #list with one array for each motion. Each array contains the normalized power for each motion
    correlation_list = []
    motion_list = []
    
    baselinelist_with_motions = {}
    correlationlist_with_motions = {}
    for bf_frame in baseline_frame:
        baseline_array = []
        baseline_df = baseline_frame[bf_frame]
        motion_list.append(baseline_frame[bf_frame]['Motion Name'])
        # if 'Motion Name' in baseline_df :
        #     del baseline_df['Motion Name']
        base = (baseline_df.values())
        for item in base:
            try:
                baseline_array.append(float(item[0]))
            except ValueError:
                continue
        bl_array =np.array(baseline_array)
        baseline_list.append(bl_array)
        
    for motion in motion_list:
        baselinelist_with_motions[motion]= []
        correlationlist_with_motions[motion]= []
    for i in range(0,len(baseline_list)):
        baselinelist_with_motions[motion_list[i]].append(baseline_list[i])
    for key in baselinelist_with_motions:
        motion_list_Not_repeated.append(key)
#calculating correlation for each frame in test frame with the frames in baseline frame list

    number =0
    for frame in test_frames:
        correlation_matrix = []
        test_frame_array = []
        test_df = test_frames[frame]
        true_interval_label.append(test_frames[frame]['Motion Name'])
        # if 'Motion Name' in test_df:
        #     del test_df['Motion Name']
        test_values = (test_df.values())
        for item in test_values:
            try:
                test_frame_array.append(float(item[0]))
            except ValueError:
                continue
        tf_array =np.array(test_frame_array)
        
        
        for key in baselinelist_with_motions:
            correlation_per_key = []
            for df in baselinelist_with_motions[key]:
                covariance = cov(df, tf_array)
                
                from statistics import variance
                var_test = variance(tf_array)
                var_baseline = variance(df)
                
                correlation = covariance/(math.sqrt((var_test*var_baseline)))
                correlation_per_key.append(correlation)
            avg_correlation_per_key = sum(correlation_per_key)/(len(correlation_per_key))
            correlationlist_with_motions[key]= avg_correlation_per_key 
            

#Calculating Interval Label: First initialize max and index variable. then iterate over for loop over all the items in list
        max_value = max(correlationlist_with_motions, key=correlationlist_with_motions.get)
        # predicted_interval_label.append([max_value])
        
#         #Correction Algorithm
        
        correction_list.append([max_value]) 
        predicted_interval_label.append(max_value)   #If you do not want to use the correction algorithm, then comment if...predicted_interval_label.append(correction_list[0][0]) and  predicted_interval_label.append(correction_list[1][1])
#         #correction list for dataframe1 is a list of one list with integer, for dataframe2 it should be a list that has 2 lists with one integer
#         #after dataframe2, correction list has two lists, each list has 3 values: uncorrected label, corrected label, and correction index

        if frame=="dataframe2":
            if correction_list[0][0] == correction_list[1][0]:
                correction_list[1].append(correction_list[1][0])
                correction_list[1].append(0)
            elif correction_list[0][0] != correction_list[1][0]: 
                correction_list[1].append(correction_list[0][0])
                correction_list[1].append(1)
        elif frame !="dataframe2" and frame !="dataframe1":
            if correction_list[0][1] == correction_list[1][0]:
                correction_list[1].append(correction_list[1][0])
                correction_list[1].append(0)
            elif correction_list[0][1] != correction_list[1][0] and correction_list[0][2] == 0: 
                correction_list[1].append(correction_list[0][1])
                correction_list[1].append(1)
            elif correction_list[0][1] != correction_list[1][0] and correction_list[0][2] == 1:
                correction_list[1].append(correction_list[1][0])
                correction_list[1].append(0)
        

                
        
#         if frame=="dataframe1":
#             predicted_interval_label.append(correction_list[0][0])
            
# #             if correction_list[0][0] == 0:
# #                 predicted_motion = motion_list[0]
# #             elif correction_list[0][0] == 1:
# #                 predicted_motion = motion_list[1]
# #             elif correction_list[0][0] == 2:
# #                 predicted_motion = motion_list[2]
# #             elif correction_list[0][0] == 3:
# #                 predicted_motion = motion_list[3]
# #             elif correction_list[0][0] == 4:
# #                 predicted_motion = motion_list[4]
# #             elif correction_list[0][0] == 5:
# #                 predicted_motion = motion_list[5]
# #             predicted_interval_label.append(predicted_motion)

            
#         else:
#             predicted_interval_label.append(correction_list[1][1])
# #             if correction_list[1][1] == 0:
# #                 predicted_motion = motion_list[0]
# #             elif correction_list[1][1] == 1:
# #                 predicted_motion = motion_list[1]
# #             elif correction_list[1][1] == 2:
# #                 predicted_motion = motion_list[2]
# #             elif correction_list[1][1] == 3:
# #                 predicted_motion = motion_list[3]
# #             elif correction_list[1][1] == 4:
# #                 predicted_motion = motion_list[4]
# #             elif correction_list[1][1] == 5:
# #                 predicted_motion = motion_list[5]
#             # predicted_interval_label.append(predicted_motion)
            
            
            
            
#         #     print(correction_list)
        
        
#         correlation_list.append(correlation_matrix) #list with each test frame's correlation numbers
        
#         test_list.append(tf_array)
        
#     # for i in range(0,int(test_list)):
        
#         # for key,value in test_df.items():
#         #     if 'EMG' in key and frame == "dataframe1":
#         #         channel_counter += 1
#       # number_of_frames = len(test_frames)/channel_counter
                
#             # # print(key, value)
#             # for bl_frame in baseline_frame:
#             #     baseline_df = baseline_frame[bl_frame]
#             #     for baseline_key, baseline_value in baseline_df.items():
#             #         if baseline_key == key:
#             #             baseline_key = key
#     # return(baseline_frame,  test_list, correlation_list, motion_list )
    # return(baselinelist_with_motions, baseline_list, test_frames, correlationlist_with_motions, max_value) 
    return(baseline_frame,  test_frames, baseline_list, motion_list_Not_repeated, predicted_interval_label, true_interval_label , correction_list)


correlate = cross_correlation(baseline,test[0])
predicted_interval_label = correlate[4]
true_interval_label = correlate[5]
display_labels= correlate[3]

from sklearn import metrics

results = metrics.accuracy_score(true_interval_label, predicted_interval_label)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

cm = confusion_matrix(true_interval_label, predicted_interval_label)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=1.5)
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=60)
# ax.tick_params(labelsize=20)  
plt.show(block=False)


# disp.plot()
#Confusion Matrix for Random Forrest
# label_font = {'size':'18'}  # Adjust to fit


# cm = confusion_matrix(correlate[5], correlate[4])
# disp = ConfusionMatrixDisplay.from_predictions(correlate[5], correlate[4],  display_labels=correlate[3],  xticks_rotation="vertical")
# total = cm.diagonal()/cm.sum(axis=1)


# # disp.plot()
# # for labels in disp.text_.ravel():
# #     labels.set_fontsize(30)
# # plt.title('Interval Motion Confusion Matrix ')
# # plt.show()



    
    