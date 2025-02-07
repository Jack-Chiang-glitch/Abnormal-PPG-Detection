import pandas as pd
import numpy as np
from scipy.signal import firwin,filtfilt
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
from scipy.signal import butter, filtfilt

def Remove_outliers(ppg_sequence):
    Q1 = np.quantile(ppg_sequence, 0.25)
    # 計算第 3 四分位數 (Q3, 0.75 分位數)
    Q3 = np.quantile(ppg_sequence, 0.75)
    IQR = Q3 - Q1
    # 定義離群值範圍
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (ppg_sequence < lower_bound) | (ppg_sequence > upper_bound)
    for i in range(len(outliers)):
        if outliers[i] == True:
            ppg_sequence[i] = ppg_sequence[i-1]      
    return ppg_sequence

def Filtering(ppg_sequence):
    fs=100
    lower_freq_bound = 0.5
    upper_freq_bound = 5
    
    
    taps = firwin(300, [lower_freq_bound / (fs / 2), upper_freq_bound / (fs / 2)], pass_zero=False)
    ppg_sequence = filtfilt(taps,1, np.double(ppg_sequence))
    return ppg_sequence

def PPG_FR_processing(ppg_sequence):
    ppg_sequence = Filtering(ppg_sequence)
    ppg_sequence = Remove_outliers(ppg_sequence)
    return ppg_sequence

def PPG_RF_processing(ppg_sequence):
    ppg_sequence = Remove_outliers(ppg_sequence)
    ppg_sequence = Filtering(ppg_sequence)
    return ppg_sequence

def Get_raw_ppg(i,j):
    # 設定主資料夾路徑
    main_directory = "C:\\Jack\\Rppg_acquire_8.18\\The-rPPG-Acquiring-System-main\\TriAnswer_Records(unprocessed)"
    folder_name = f"{str(i).zfill(2)}-{str(j).zfill(2)}"
    #ile_path = os.path.join(main_directory, folder_name, "PPG_R.csv")
    file_path = os.path.join(main_directory, folder_name, "PPG_R.csv")
    # 讀取 CSV 檔案
    ppg_data_frame = pd.read_csv(file_path, header=None)
    # 取得第一個欄位並轉為 numpy 陣列
    raw_ppg = ppg_data_frame.iloc[:, 0].to_numpy() # get first 
    raw_ppg = raw_ppg[:30000]
    
    return raw_ppg

def Save_ppg_dataframe(ppg_dataframe, i, j):  # R : Remove outliers F : Filtering 1 : intitial 2 : final
    main_directory = "C:\\Jack\\Rppg_acquire_8.18\\The-rPPG-Acquiring-System-main\\TriAnswer_Records(processed)"
    folder_name = f"{str(i).zfill(2)}-{str(j).zfill(2)}"
    output_file_path = os.path.join(main_directory, folder_name, "PPG_processed.csv")
    ppg_dataframe.to_csv(output_file_path, index=False)
    #np.savetxt(output_file_path, ppg_dataframe, delimiter=',', header='PPG Data', comments='', fmt='%.6f')




def Retrieve_tester_list():
    # 生成範圍1到15的數字，排除7和10
    range_1_to_15 = np.setdiff1d(np.arange(1, 16), [7, 10])
    # 生成範圍51到58的數字
    range_51_to_58 = np.arange(51, 59)
    # 合併兩個範圍
    result = np.concatenate((range_1_to_15, range_51_to_58))
    return [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,   \
            11,12,13,14,15,16,17,18,19,20,   \
            21,22,23,24,25,26,27,28,29,30,   \
            31,32,33,34,35,36,37,38,39,40,41, \
            
            51,52,53,54,55,56,57,58,          \
            61,   63,64,   66,67,68,   70]



def Clip_2_std(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    
    # 計算上下界 (平均值 ± 2 * 標準差)
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    
    # Clip 訊號
    clipped_signal = np.clip(signal, lower_bound, upper_bound)
    return clipped_signal


# 1. 巴特沃斯濾波器 (0.5 Hz - 8 Hz)
def butter_bandpass_filter(data, low_cut, high_cut, sampling_rate, order=2):
    nyquist = 0.5 * sampling_rate
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)



# 建立 01-01 到 15-04 的資料夾
for i in Retrieve_tester_list():
    for j in range(1, 5):
        if i == 7 and j == 1:
            continue
        raw_ppg = Get_raw_ppg(i,j)
        
        #FR1_ppg = PPG_FR_processing(np.copy(raw_ppg))
        #FR2_ppg = PPG_FR_processing(np.copy(FR1_ppg))
        
        #RF1_ppg = PPG_RF_processing(np.copy(raw_ppg))
        #RF2_ppg = PPG_RF_processing(np.copy(RF1_ppg))
        
        clipped_signal = Clip_2_std(np.copy(raw_ppg))
        #neurokit_ppg = nk.ppg_clean(clipped_signal, sampling_rate=100)
        neurokit_ppg = butter_bandpass_filter(clipped_signal, low_cut=0.5, high_cut=8.0, sampling_rate=100)
        
        
        # 將這四個陣列組合成一個 DataFrame，並指定欄位名稱
        ppg_dataframe = pd.DataFrame({
            #"Raw_clipped": clipped_signal
            "Neurokit": neurokit_ppg,
            #"FR1":      FR1_ppg,
            #"FR2":      FR2_ppg,
            #"RF1":      RF1_ppg,
            #"RF2":      RF2_ppg
        })
        Save_ppg_dataframe(ppg_dataframe, i, j)
        print(f"{str(i).zfill(2)}-{str(j).zfill(2)}")
    print()




