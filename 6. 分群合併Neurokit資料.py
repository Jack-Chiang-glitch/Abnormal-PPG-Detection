import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ppg_library import Retrieve_tester_list
from ppg_library import split_into_chunks




# 主資料夾路徑
#main_directory = "C:\\Jack\\Rppg_acquire_8.18\\The-rPPG-Acquiring-System-main\\TriAnswer_Records(processed)\\Neurokit_process"
main_directory = "C:\\Jack\\Rppg_acquire_8.18\\The-rPPG-Acquiring-System-main\\TriAnswer_Records(processed)\\FR1_process"

def GetFirstRest(i):
    serial_number = f"{str(i).zfill(2)}"
    file_path = os.path.join(main_directory, serial_number + "_PPG.csv")
    ppg_dataframe = pd.read_csv(file_path)
    return ppg_dataframe["First_rest"]

def GetPsycho(i):
    serial_number = f"{str(i).zfill(2)}"
    file_path = os.path.join(main_directory, serial_number + "_PPG.csv")
    ppg_dataframe = pd.read_csv(file_path)
    return ppg_dataframe["Psycho"]



def Leave_one_subject_out_acc(removed_num):
    train_tester_list = Retrieve_tester_list[Retrieve_tester_list!=removed_num]
    
    train_FirstRest, train_Psycho = np.array([GetFirstRest(i) for i in train_tester_list]), \
                                    np.array([GetPsycho(i)    for i in train_tester_list])
    
    test_FirstRest, test_Psycho = GetFirstRest(removed_num).to_numpy(), \
                                  GetPsycho(removed_num).to_numpy()
    
    train_FirstRest, train_Psycho, test_FirstRest, test_Psycho  = \
                                split_into_chunks(train_FirstRest.flatten(), chunk_size), \
                                split_into_chunks(train_Psycho.flatten(), chunk_size), \
                                split_into_chunks(test_FirstRest.flatten(), chunk_size), \
                                split_into_chunks(test_Psycho.flatten(), chunk_size) 
    

    train_set, test_set = np.vstack((train_FirstRest, train_Psycho)), \
                          np.concatenate((test_FirstRest,  test_Psycho))
    train_labels, test_labels = np.concatenate((np.zeros(train_FirstRest.shape[0]), np.ones(train_Psycho.shape[0]))), \
                                np.concatenate((np.zeros(test_FirstRest.shape[0]), np.ones(test_Psycho.shape[0])))
    """                         
    print(train_set.shape)
    print(test_set.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    """
    
    model = SVC(kernel='rbf')  # 你可以選擇不同的核函數
    model.fit(train_set, train_labels)
    pred_labels = model.predict(test_set)
    accuracy = accuracy_score(test_labels, pred_labels)
    
    print(str(removed_num) + f' : accuracy {accuracy:.2f}')
    return accuracy
    
    """
    print(train_FirstRest.shape)
    print(train_Psycho.shape)
    print(test_FirstRest.shape)
    print(test_Psycho.shape)
    """
    

Retrieve_tester_list = Retrieve_tester_list()
list_len = len(Retrieve_tester_list)
chunk_size = 500

acc_list = np.array([Leave_one_subject_out_acc(i) for i in Retrieve_tester_list])
print("Final Accuracy : " + str(acc_list.mean()))









