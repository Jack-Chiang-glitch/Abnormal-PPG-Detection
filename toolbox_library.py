import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import random
import torch
import torch.nn.functional as F
import os
import scipy.fft
import neurokit2 as nk
import warnings
from numpy.lib.stride_tricks import sliding_window_view

from model_library import Autoencoder


class LF_RF:
    @staticmethod
    def one_sided_fft(signals, fs):
        is_one_dim = (signals.ndim == 1)
        
        if is_one_dim:
            signals = np.expand_dims(signals, axis=0)
        
        signal_len = signals.shape[1]
        frequencies = np.fft.rfftfreq(signal_len, 1/fs)
        rfft_result = np.fft.rfft(signals)
        amplitudes = np.abs(rfft_result) * 2 / signal_len
        amplitudes[:, 0] /= 2
        if signal_len % 2 == 0:
            amplitudes[:, -1] /= 2
            
        if is_one_dim:
            amplitudes = amplitudes.squeeze()
        return frequencies, amplitudes

    @staticmethod
    def power_spectrum(amplitudes):
        return (amplitudes**2) / 2

    @staticmethod
    def calc_LF_RF_ratio(freq, power_spectrum):
        LF_idx = (freq >= 0.05) & (freq <= 0.15)
        HF_idx = (freq >= 0.15) & (freq <= 0.4)
        LF = power_spectrum[:, LF_idx].sum(axis=1)
        HF = power_spectrum[:, HF_idx].sum(axis=1)
        return LF / HF

    @staticmethod
    def LF_RF_ratio(signals, fs):
        freq, amplitudes = LF_RF.one_sided_fft(signals, fs=fs)  # 修正方法名稱
        power_spectrum = LF_RF.power_spectrum(amplitudes)  # 修正方法名稱
        LF_RF_ratio = LF_RF.calc_LF_RF_ratio(freq, power_spectrum)  # 修正方法名稱
        return LF_RF_ratio
        
    


class ToolBox:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, chunk_size=400):
        self.chunk_size = chunk_size       
    def split_into_chunks(self, array):
        return array.reshape(-1, self.chunk_size)

    @staticmethod
    def rfft_with_cut_freq(np_signals, cut_freq=8, fs=100):
        N = len(np_signals[0])
        
        torch_signals = torch.from_numpy(np_signals)
        fft_result = torch.fft.rfft(torch_signals)  # 直接輸出正頻率部分
        freqs = torch.fft.rfftfreq(N, d=1/fs)  # 正頻率對應的頻率軸
        mask = freqs <= cut_freq
        return fft_result[:, mask]
        
    
    @staticmethod
    def safe_ppg_quality(ppg_signal, sampling_rate=100):
        try:
            # 嘗試計算信號品質
            quality = nk.ppg_quality(ppg_signal, sampling_rate=sampling_rate)
            return np.mean(quality)  # 返回品質平均值
        except Exception as e:
            # 如果出錯，返回品質為 0（代表不合格）
            return 0
        
    @staticmethod
    def FFT_numpy(signals, fs=100):
        # If input is 1D, reshape it to 2D with one row
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
    
        # Perform FFT along each row
        
        full_spectrum = scipy.fft.fft(signals, axis=1)  # FFT along rows
        return full_spectrum
        
    @staticmethod
    def one_sided_FFT_numpy(signals, fs=100):
        N = signals.shape[1]
        full_spectrum = ToolBox.FFT_numpy(signals, fs=100)
        return full_spectrum[:, :N // 2 + 1]

    @staticmethod
    def one_sided_to_full_spectrum_torch(half_y_f):
        #print(half_y_f[0])
        #print(torch.conj(half_y_f[:, 1:-1].flip(dims=[1]))[0])
        #print(torch.cat((half_y_f, torch.conj(half_y_f[:, 1:-1].flip(dims=[1]))), dim=1)[0])
        return torch.cat((half_y_f, torch.conj(half_y_f[:, 1:-1].flip(dims=[1]))), dim=1)


        
    @staticmethod
    def IDFT_torch(y_f):
        time_signal = torch.fft.ifft(y_f).real  # Restore the time-domain signal (real part)
        return time_signal
        
        
    @staticmethod
    def DFT(signal_1d, fs=100):  # Discrete_Fourier_Transform
        N = len(signal_1d)
        y_f = scipy.fft.fft(signal_1d)              # 離散傅立葉轉換
        x_f = np.linspace(0.0, fs/2, N // 2)        # 只取正頻率部分
        amplitude = 2.0 / N * np.abs(y_f[: N // 2]) # 計算幅度並正規化
        freq_mask = (x_f >= 0) & (x_f <= 5)         # 獲取 0-5 Hz 的索引
        magnitudes = amplitude[freq_mask]           # 獲取頻率和對應的幅度向量
        return magnitudes

    @staticmethod
    def extract_DFT_features(array):
        return np.array([ToolBox.DFT(signal) for signal in array]) 

    @staticmethod
    def get_labels(num_seg, mode): # num_rest_seg = 休息的PPG有幾段，以此類推
        if mode == 'rest':
            rest_labels   = np.array([0] * num_seg)
            return rest_labels

        if mode == 'psycho':
            psycho_labels = np.array([1] * num_seg)
            return psycho_labels


    @staticmethod
    def load_pretrained_autoencoder():  # 靜態方法
        checkpoint_path = "autoencoder_ppg_best.ckpt"
        try:
            model = Autoencoder("encode")
            model = model.to(ToolBox.device)
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            model.eval()
            return model
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            raise

    @staticmethod
    def extract_autoencoder_features(array):   # 輸入是 np.array
        array = array.astype(np.float32)
        array = torch.from_numpy(array).to(ToolBox.device)
        autoencoder = ToolBox.load_pretrained_autoencoder()
        with torch.no_grad():
           output = autoencoder(array)
        return output

    @staticmethod
    def get_repeated_ppg_volatility(array, repeated_times, window_size = 100): # 2d nparray
        
        # 創建滑動窗口，形狀為 (條數, 窗口數, 窗口大小)
        windows = sliding_window_view(array, window_shape=window_size, axis=1)
        
        # 計算滑動標準差，結果形狀為 (條數, 窗口數)
        std_results = np.std(windows, axis=-1)
        
        # 將結果填充回與原始數據相同大小的陣列
        padded_std_results = np.full(array.shape, np.nan)
        padded_std_results[:, window_size-1:] = std_results
        
        padded_std_results = padded_std_results[:, ~np.isnan(padded_std_results).all(axis=0)]


        return np.repeat(padded_std_results, repeated_times, axis=0)

    @staticmethod
    def get_ppg_volatility(array, window_size = 100): # 2d nparray
        
        # 創建滑動窗口，形狀為 (條數, 窗口數, 窗口大小)
        windows = sliding_window_view(array, window_shape=window_size, axis=1)
        
        # 計算滑動標準差，結果形狀為 (條數, 窗口數)
        std_results = np.std(windows, axis=-1)
        
        # 將結果填充回與原始數據相同大小的陣列
        padded_std_results = np.full(array.shape, np.nan)
        padded_std_results[:, window_size-1:] = std_results
        
        padded_std_results = padded_std_results[:, ~np.isnan(padded_std_results).all(axis=0)]


        return padded_std_results





class DataAugmentation:  
    def __init__(self, chunk_size=400, shift_step=100):
        self.chunk_size = chunk_size 
        self.shift_step = shift_step
    
    def time_series_shift(self,array):
        # array size = [N, 30000] N = num of testers
        # 待會會刪除掉 self.chunk_size 那麼多個column
        # 所以建立一個column有 30000 - self.chunk_size的空陣列
        arr_len = array.shape[1]
        #print( arr_len - self.chunk_size)
        shifted_array = np.empty((0, arr_len - self.chunk_size))
        for num in range(0, int(self.chunk_size / self.shift_step)): # 位移分別是 0, 100, 200, 300
            shift = self.shift_step * num
            num_cols_to_remove_front = shift                      # 前面要去除的行數
            num_cols_to_remove_end   = self.chunk_size - shift    # 後面要去除的行數
            removed_col_array = array[:, num_cols_to_remove_front : -num_cols_to_remove_end]
            shifted_array = np.vstack((shifted_array, removed_col_array))
        return shifted_array
    """
    eg.
    # augmentor = DataAugmentation(chunk_size=8, shift_step=2)
    # arr = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    #                  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,14., 15., 16.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,1.,  1.,  1.],
           
           [ 3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.,16., 17., 18.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,1.,  1.,  1.],
           
           [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,1.,  1.,  1.],
           
           [ 7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.],
           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])

    """

    def add_nosie(self,array, mean=0, std=1):
        # array size = [N, 30000] N = num of testers
        noise = np.random.normal(mean, std, array.shape)
        return array+noise # size [N, 30000]




class Unqualified_PPG_preprocessor:
    warnings.filterwarnings("ignore", message="Too few peaks detected to compute the rate")
    def __init__(self, data_dir='../train', compare_type='Psycho', chunk_size=400, 
                 shift_step=100, get_cleaned_data=False, calculate_quality=False):
        self.save_dir = data_dir
        
        # preprocessing tool allocation
        augmentor = DataAugmentation(chunk_size, shift_step)
        tool      = ToolBox(chunk_size)
        # directory
        all_csv_filepath_list = self.all_csv_filepath_list(data_dir)
        
        
        self.raw_rest    = self.raw_PPG(all_csv_filepath_list, "First_rest") # size = N x 30000 (N = number of testers) <class 'numpy.ndarray'>
        self.raw_psycho  = self.raw_PPG(all_csv_filepath_list, compare_type)     # size = N x 30000 (N = number of testers) <class 'numpy.ndarray'>
        shifted_rest   = augmentor.time_series_shift(self.raw_rest)    # size = [4 x N, 30000-chunk_size] <class 'numpy.ndarray'>  乘以 4 因為 shift = 0, 100, 200, 300 
        shifted_psycho = augmentor.time_series_shift(self.raw_psycho)  # size, class 同上，shift_rest和shift_psycho包含了位移以及未位移的PPG資料

        self.chunked_rest   = tool.split_into_chunks(shifted_rest)   # size = [M, chunk_size] <class 'numpy.ndarray'> where M * chunk_size = 4N * (30000-chunk_size)
        self.chunked_psycho = tool.split_into_chunks(shifted_psycho) # size = [M, chunk_size] <class 'numpy.ndarray'>

        
        whole_ppg = np.vstack((self.raw_rest, self.raw_psycho))
        repeated_times = len(self.chunked_rest) / len(self.raw_rest)
        self.ppg_volatility = tool.get_repeated_ppg_volatility(whole_ppg[ : , :1100], 
                                                      repeated_times, window_size = 100) #前面1分鐘 多100是因為損耗100個點計算ppg
        """
        print(self.ppg_volatility.shape)
        print(f"raw rest : {self.raw_rest.shape}")
        print(f"raw psycho : {self.raw_psycho.shape}")
        print(f"chunked rest : {self.chunked_rest.shape}")
        print(f"chunked psycho : {self.chunked_psycho.shape}")
        """

        
        if calculate_quality:
            chunked_rest_quality = np.array([self.safe_ppg_quality(ppg)
                                             for ppg in self.chunked_rest])
            chunked_psycho_quality = np.array([self.safe_ppg_quality(ppg)
                                               for ppg in self.chunked_psycho])
            self.save_ppg_quality_to_npy(chunked_rest_quality, chunked_psycho_quality)

        if get_cleaned_data :
            # 使用列表生成式，結合封裝的函數
            self.cleaned_chunked_rest = np.array([
                ppg_signal
                for ppg_signal in chunked_rest
                if self.safe_ppg_quality(ppg_signal, sampling_rate=100) >= 0.99
            ])
    
            self.cleaned_chunked_psycho = np.array([
                ppg_signal
                for ppg_signal in chunked_psycho
                if self.safe_ppg_quality(ppg_signal, sampling_rate=100) >= 0.99
            ])
    
            self.dict = {'First_rest':   self.cleaned_chunked_rest, 
                         'Psycho':       self.cleaned_chunked_psycho}

        
    def save_ppg_quality_to_npy(self, chunked_rest_quality, chunked_psycho_quality):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "chunked_rest_quality.npy")
        np.save(path, chunked_rest_quality)
        path = os.path.join(self.save_dir, "chunked_psycho_quality.npy")
        np.save(path, chunked_psycho_quality)
        print('saved')
    
    def save_cleaned_nparray_to_npy(self):
        # 確保路徑存在，否則創建該目錄
        os.makedirs(self.save_dir, exist_ok=True)
        
        datatypes = ['First_rest','Psycho']
        for datatype in datatypes:
            save_filename = f"{datatype}_cleaned_chunked.npy"      
            full_path = os.path.join(self.save_dir, save_filename)
            # 保存為 .npy 文件
            np.save(full_path, self.dict[datatype])

            print(f"數據已保存到: {full_path}")

    def safe_ppg_quality(self, ppg_signal, sampling_rate=100):
        try:
            # 嘗試計算信號品質
            quality = nk.ppg_quality(ppg_signal, sampling_rate=sampling_rate)
            return np.mean(quality)  # 返回品質平均值
        except Exception as e:
            # 如果出錯，返回品質為 0（代表不合格）
            return 0
            
    def all_csv_filepath_list(self, data_dir):
        return [ os.path.join(data_dir, file_name)   # 將資料夾路徑和CSV檔案連結在一起，形成完整路徑
                 for file_name in os.listdir(data_dir)
                 if file_name.endswith('.csv')] # 迭代資料夾裡所有的檔案(CSV檔)
        # 得到完整的路徑list 例如[./.../a.csv,  ./.../b.csv, ./.../c.csv]
    def raw_PPG(self, all_csv_filepath_list, stress_type="First_rest"):
        # df = pd.read_csv(filepath)是一個 dataframe
        # 只是因為lamda裡面不能新設立一個變數，所以出此下策，例如df["Psycho"].values
        # PPG_per_tester return vector type = <class 'numpy.ndarray'>
        # shape = (30000,)
        PPG_per_tester = lambda filepath : pd.read_csv(filepath)[stress_type].values 
        return np.array([PPG_per_tester(filepath) 
                         for filepath in all_csv_filepath_list])











class Unqualified_PPG_preprocessor2:
    warnings.filterwarnings("ignore", message="Too few peaks detected to compute the rate")
    def __init__(self, data_dir='../train', compare_type='Psycho', chunk_size=400, 
                 shift_step=100, get_cleaned_data=False, calculate_quality=False):
        self.save_dir = data_dir
        
        # preprocessing tool allocation
        augmentor = DataAugmentation(chunk_size, shift_step)
        tool      = ToolBox(chunk_size)
        # directory
        all_csv_filepath_list = self.all_csv_filepath_list(data_dir)
        
        
        self.raw_rest    = self.raw_PPG(all_csv_filepath_list, "First_rest") # size = N x 30000 (N = number of testers) <class 'numpy.ndarray'>
        self.raw_psycho  = self.raw_PPG(all_csv_filepath_list, compare_type)     # size = N x 30000 (N = number of testers) <class 'numpy.ndarray'>



        """
        def zscore_along_rows(data):
            # 計算每行的平均值和標準差
            means = np.mean(data, axis=1, keepdims=True)
            stds = np.std(data, axis=1, keepdims=True)
            
            # 進行z-score標準化
            normalized_data = (data - means) / stds
            return normalized_data
        
        self.raw_rest   = zscore_along_rows(self.raw_rest)  # shape仍然是(45, 30000)
        self.raw_psycho = zscore_along_rows(self.raw_psycho)  # shape仍然是(45, 30000)
        """


        
        shifted_rest   = augmentor.time_series_shift(self.raw_rest)    # size = [4 x N, 30000-chunk_size] <class 'numpy.ndarray'>  乘以 4 因為 shift = 0, 100, 200, 300 
        shifted_psycho = augmentor.time_series_shift(self.raw_psycho)  # size, class 同上，shift_rest和shift_psycho包含了位移以及未位移的PPG資料

        self.chunked_rest   = tool.split_into_chunks(shifted_rest)   # size = [M, chunk_size] <class 'numpy.ndarray'> where M * chunk_size = 4N * (30000-chunk_size)
        self.chunked_psycho = tool.split_into_chunks(shifted_psycho) # size = [M, chunk_size] <class 'numpy.ndarray'>

        """
        whole_ppg = np.vstack((self.raw_rest, self.raw_psycho))
        repeated_times = len(self.chunked_rest) / len(self.raw_rest)
        self.ppg_volatility = tool.get_ppg_volatility(whole_ppg[ : , :1100], 
                                                      repeated_times, window_size = 100) #前面1分鐘 多100是因為損耗100個點計算ppg
        """

        
    def save_ppg_quality_to_npy(self, chunked_rest_quality, chunked_psycho_quality):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "chunked_rest_quality.npy")
        np.save(path, chunked_rest_quality)
        path = os.path.join(self.save_dir, "chunked_psycho_quality.npy")
        np.save(path, chunked_psycho_quality)
        print('saved')
    
    def save_cleaned_nparray_to_npy(self):
        # 確保路徑存在，否則創建該目錄
        os.makedirs(self.save_dir, exist_ok=True)
        
        datatypes = ['First_rest','Psycho']
        for datatype in datatypes:
            save_filename = f"{datatype}_cleaned_chunked.npy"      
            full_path = os.path.join(self.save_dir, save_filename)
            # 保存為 .npy 文件
            np.save(full_path, self.dict[datatype])

            print(f"數據已保存到: {full_path}")

    def safe_ppg_quality(self, ppg_signal, sampling_rate=100):
        try:
            # 嘗試計算信號品質
            quality = nk.ppg_quality(ppg_signal, sampling_rate=sampling_rate)
            return np.mean(quality)  # 返回品質平均值
        except Exception as e:
            # 如果出錯，返回品質為 0（代表不合格）
            return 0
            
    def all_csv_filepath_list(self, data_dir):
        return [ os.path.join(data_dir, file_name)   # 將資料夾路徑和CSV檔案連結在一起，形成完整路徑
                 for file_name in os.listdir(data_dir)
                 if file_name.endswith('.csv')] # 迭代資料夾裡所有的檔案(CSV檔)
        # 得到完整的路徑list 例如[./.../a.csv,  ./.../b.csv, ./.../c.csv]
    def raw_PPG(self, all_csv_filepath_list, stress_type="First_rest"):
        # df = pd.read_csv(filepath)是一個 dataframe
        # 只是因為lamda裡面不能新設立一個變數，所以出此下策，例如df["Psycho"].values
        # PPG_per_tester return vector type = <class 'numpy.ndarray'>
        # shape = (30000,
        
        PPG_per_tester = lambda filepath : pd.read_csv(filepath)[stress_type].values 
        return np.array([PPG_per_tester(filepath) 
                         for filepath in all_csv_filepath_list])

