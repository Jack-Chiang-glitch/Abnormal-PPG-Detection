import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import torch
import torch.nn.functional as F
import os
import scipy.fft
import neurokit2 as nk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from toolbox_library import DataAugmentation, ToolBox, Unqualified_PPG_preprocessor, LF_RF, Unqualified_PPG_preprocessor2


class Clipped_PPGDataset(Dataset):
    def __init__(self, data_dir='../train'):
        def load_data(file_name, is_list_object=False):
            if is_list_object:
                return np.load(os.path.join(data_dir, file_name), allow_pickle=True).tolist()
            else:
                np_array = np.load(os.path.join(data_dir, file_name))
                return torch.tensor(np_array, dtype=torch.float32)
                

        # 加載數據
        self.chunked_ppgs       = load_data('chunked_ppgs.npy')
        self.one_sided_reals    = load_data('one_sided_reals.npy')
        self.one_sided_imags    = load_data('one_sided_imags.npy')
        self.half_spectrums     = torch.complex(self.one_sided_reals, self.one_sided_imags)
        self.ppg_quality_vecs   = load_data('ppg_quality_vectors.npy')
        self.ppg_scores         = load_data('ppg_scores.npy')
        self.normalized_amplis  = load_data('normalized_amplitudes.npy')
        self.normal_peaks_list  = load_data('normal_peaks_list.npy', is_list_object=True)
        self.reverse_peaks_list = load_data('reverse_peaks_list.npy', is_list_object=True)


        _MAXs = torch.max(abs(self.chunked_ppgs), dim=1).values.unsqueeze(1)
        self.chunked_ppgs = 200 * self.chunked_ppgs / _MAXs
        self.half_spectrums = torch.fft.rfft(self.chunked_ppgs)
        """
        _amplitudes = torch.abs(self.half_spectrums)
        _MAXs = torch.max(_amplitudes, dim=1).values.unsqueeze(1)
        self.half_spectrums = 20* self.half_spectrums / _MAXs 

        _full_spectrums = ToolBox.one_sided_to_full_spectrum_torch(self.half_spectrums)
        self.chunked_ppgs = torch.fft.ifft(_full_spectrums).real
        """
        
        # 過濾條件：ppg_scores <= 0.8
        valid_indices = torch.where( (self.ppg_scores <=0.8) & (self.ppg_scores>=0.7) )[0]

        # 根據過濾條件篩選數據
        self.chunked_ppgs       = self.chunked_ppgs[valid_indices]
        self.half_spectrums     = self.half_spectrums[valid_indices]
        self.one_sided_reals    = self.one_sided_reals[valid_indices]
        self.one_sided_imags    = self.one_sided_imags[valid_indices]
        self.ppg_quality_vecs   = self.ppg_quality_vecs[valid_indices]
        self.ppg_scores         = self.ppg_scores[valid_indices]
        self.normalized_amplis  = self.normalized_amplis[valid_indices]
        self.normal_peaks_list  = [self.normal_peaks_list[i] for i in valid_indices]
        self.reverse_peaks_list = [self.reverse_peaks_list[i] for i in valid_indices]
        

    # 自訂的 collate_fn 函數
    @staticmethod
    def custom_collate_fn(batch):
        chunked_ppgs       = torch.stack([item[0] for item in batch])  # 將 `chunked_ppgs` 堆疊成張量
        half_spectrums     = torch.stack([item[1] for item in batch])
        normalized_amplis   = torch.stack([item[2] for item in batch])  # 將 `normalized_ampls` 堆疊成張量
        normal_peaks_list  = [item[3] for item in batch]                # 保持 `normal_peaks_list` 為原始列表
        
        return chunked_ppgs, half_spectrums, normalized_amplis, normal_peaks_list

    def __len__(self):
        return len(self.chunked_ppgs)

    def __getitem__(self, idx):
        return (
            self.chunked_ppgs[idx],
            self.half_spectrums[idx],
            self.normalized_amplis[idx],
            self.normal_peaks_list[idx],
            
            #self.one_sided_reals[idx],
            #self.one_sided_imags[idx],
            #self.ppg_quality_vecs[idx],
            #self.ppg_scores[idx],
            
            #self.reverse_peaks_list[idx],
        )

        



#################
# for example
# train_set = RawPPGDataset('../train', chunk_size=1000, shift_step=100)
# valid_set = RawPPGDataset('../valid', chunk_size=1000, shift_step=100)
#################
class RawPPGDataset(Dataset):
    def __init__(self, data_dir='../train', chunk_size=1000, shift_step=100, sampling_rate=100):
        print(data_dir+str('----------------------------------'))
        preprocessor = Unqualified_PPG_preprocessor(data_dir, chunk_size, shift_step)
        _chunked_rest   = preprocessor.chunked_rest
        _chunked_psycho = preprocessor.chunked_psycho
        _chunked_ppgs = np.vstack((_chunked_rest, _chunked_psycho))





        _chunked_ppgs, _normal_peaks_list, _reverse_peaks_list = \
                                    self.Ensure_peaks_are_not_empty(_chunked_ppgs.copy())
        
        _chunked_ppgs = self.Clip(_chunked_ppgs.copy(), _normal_peaks_list, _reverse_peaks_list)



        self.normal_peaks_list  = self.get_peaks(_chunked_ppgs)
        self.reverse_peaks_list = self.get_peaks(-1*_chunked_ppgs)

        self.ppg_quality_vecs = np.array([nk.ppg_quality(chunked_ppg, sampling_rate=100)
                                          for chunked_ppg in _chunked_ppgs])
        
        self.ppg_scores = self.ppg_quality_vecs.mean(axis=1)

        
        
        self.chunked_ppgs = torch.from_numpy(_chunked_ppgs)

        _half_spectrums = torch.fft.rfft(self.chunked_ppgs)

        _amplitude = torch.abs(_half_spectrums[ : , :80])
        _MAXs = torch.max(_amplitude, dim=1).values.unsqueeze(1)
        self.normalized_ampls = 10* _amplitude / _MAXs 
        
        self.one_sided_reals = _half_spectrums.real
        self.one_sided_imags = _half_spectrums.imag
        
        #self.Convert_datatype_to_accustom_to_model()
        self.half_spectrums = torch.complex(self.one_sided_reals, self.one_sided_imags)

        self.Save_files(data_dir)

    def Save_files(self, data_dir):
        
        np.save(os.path.join(data_dir, "normal_peaks_list.npy"), 
                np.array(self.normal_peaks_list, dtype=object))
        
        np.save(os.path.join(data_dir, "reverse_peaks_list.npy"), 
                np.array(self.reverse_peaks_list, dtype=object))

        np.save(os.path.join(data_dir, "ppg_quality_vectors.npy"),
                self.ppg_quality_vecs)

        np.save(os.path.join(data_dir, "ppg_scores.npy"),
                self.ppg_scores)

        np.save(os.path.join(data_dir, "chunked_ppgs.npy"),
                self.chunked_ppgs.numpy())

        np.save(os.path.join(data_dir, "normalized_amplitudes.npy"),
                self.normalized_ampls.numpy())

        np.save(os.path.join(data_dir, "half_spectrums.npy"),
                self.half_spectrums.numpy())

        np.save(os.path.join(data_dir, "one_sided_reals.npy"),
                self.one_sided_reals.numpy())
                
        np.save(os.path.join(data_dir, "one_sided_imags.npy"),
                self.one_sided_imags.numpy())
        
    
    

    def Ensure_peaks_are_not_empty(self, chunked_signals):
        
        
        chunked_signals = np.clip(chunked_signals, a_min=-800, a_max=800)
        normal_peaks_list = self.get_peaks(chunked_signals)
        reverse_peaks_list = self.get_peaks(-1*chunked_signals)

        count = 0
        discard_set = []
        
        
        for i in range(len(chunked_signals)):

            clip_max = 810
            while( len(normal_peaks_list[i])<=6 or  
                   len(reverse_peaks_list[i])<=6):
                if clip_max<=0:
                    #print(len(reverse_peaks_list[i]),len(normal_peaks_list[i]))
                    break
                
                clip_max-=10
                chunked_signals[i] = np.clip(chunked_signals[i], -clip_max, clip_max)

                try:
                    normal_peaks_list[i] = self.get_peak_for_single_signal(chunked_signals[i])
                    reverse_peaks_list[i] = self.get_peak_for_single_signal(-1*chunked_signals[i])
                except:
                    #print('==================================')
                    count+=1
                    discard_set.append(i)
                    #print(f"clip_max = {clip_max}")
                    #print('error '+str(i))
        print(f"count = {count}")

        chunked_signals = np.array([chunked_signals[i] for i in range(len(chunked_signals)) if i not in discard_set])
        normal_peaks_list = [normal_peaks_list[i] for i in range(len(normal_peaks_list)) if i not in discard_set]
        reverse_peaks_list = [reverse_peaks_list[i] for i in range(len(reverse_peaks_list)) if i not in discard_set]
        
        return chunked_signals, normal_peaks_list, reverse_peaks_list
                
                
            
        


    def Clip(self, chunked_signals, normal_peaks_list, reverse_peaks_list):
        def get_max(signal, peaks):
            sort_peak_heights = sorted(abs(signal[peaks]))
            return max(sort_peak_heights[:-2])
            
            
        for i in range(len(chunked_signals)):
            chunked_signal = chunked_signals[i]
            normal_peaks = normal_peaks_list[i]
            reverse_peaks = reverse_peaks_list[i]
            
            max_1 = get_max(chunked_signal, normal_peaks)
            max_2 = get_max(chunked_signal, reverse_peaks)
            MAX = max(max_1, max_2)
            chunked_signals[i]= np.clip(chunked_signals[i], -1.1*MAX, 1.1*MAX)
        return chunked_signals
            
        
        
        

    def Convert_datatype_to_accustom_to_model(self):
        self.chunked_ppgs    = self.chunked_ppgs.to(dtype=torch.float32)
        self.one_sided_reals = self.one_sided_reals.to(dtype=torch.float32)
        self.one_sided_imags = self.one_sided_imags.to(dtype=torch.float32)

        self.normalized_ampls = self.normalized_ampls.to(dtype=torch.float32)

    

    def get_peak_for_single_signal(self, signal, sampling_rate=100):
        return nk.ppg_peaks(signal, sampling_rate)[1]["PPG_Peaks"]

    def get_peaks(self, chunked_signals, sampling_rate=100):

        def _peaks(i):
            if i%12000==0:
                print(i)
            return nk.ppg_peaks(chunked_signals[i], sampling_rate)[1]["PPG_Peaks"]
            
        print(chunked_signals.shape)
        return [ _peaks(i)
                 for i in range(len(chunked_signals)) ]




class Dataset_for_auto(Dataset):
    def __init__(self, data_dir='../train', chunk_size=400, shift_step=100, threshold = 0.96):
        print('Dataset for auto')
                
        preprocessor = Unqualified_PPG_preprocessor(data_dir=data_dir, chunk_size=chunk_size)
        
        rest_qualities   = np.load(os.path.join(data_dir, 'chunked_rest_quality.npy'))
        psycho_qualities = np.load(os.path.join(data_dir, 'chunked_psycho_quality.npy'))
        
        rest_ppgs    = preprocessor.chunked_rest
        psycho_ppgs  = preprocessor.chunked_psycho

        
        
        rest_ppgs = self.get_ppgs_whose_quality_bigger_than_threshold(rest_ppgs, rest_qualities, threshold)
        psycho_ppgs = self.get_ppgs_whose_quality_bigger_than_threshold(psycho_ppgs, psycho_qualities, threshold)
        
        common_len = min(len(rest_ppgs), len(psycho_ppgs))

        
        self.ppgs   = np.vstack((rest_ppgs[ : common_len],     # size = [N1+N2, chunk_size] <class 'numpy.ndarray'>
                                 psycho_ppgs[ : common_len]))
        
        self.Convert_datatype_to_accustom_to_model()

    def get_ppgs_whose_quality_bigger_than_threshold(self, ppgs, qualities, threshold):
        
        # 使用 argsort 獲取從大到小的排序索引
        sorted_indices = np.argsort(qualities)[::-1]
        # 根據排序後的索引對 qualities 和 ppgs 進行排序
        qualities = qualities[sorted_indices]
        ppgs = ppgs[sorted_indices]

        mask = qualities>=threshold
        return ppgs[mask]
        

        
    def Convert_datatype_to_accustom_to_model(self):
        # 將數據類型轉換為 Tensor
        self.ppgs = torch.tensor(self.ppgs, dtype=torch.float32)
        #self.chunked_rest = torch.tensor(self.chunked_rest, dtype=torch.float32)
        #self.chunked_psycho = torch.tensor(self.chunked_psycho, dtype=torch.float32)
        
        
        
        
    def __len__(self):
        return len(self.ppgs)  # len(self.labeled_ppg) = self.labeled_ppg的row數量
        
    def __getitem__(self, idx):
        return self.ppgs[idx]
                                                            



class PCA_dataset(Dataset):
    def __init__(self, data_dir='../train', normalize = False):
        
        def standardize_sequences(data):
            standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            return standardized_data
        
        self.ppgs_pca = None
        self.labels = None
        
        train_set = PPGDataset(data_dir = '../train')
        valid_set = PPGDataset(data_dir = '../valid')

        train_ppgs = train_set.ppgs.numpy()
        valid_ppgs = valid_set.ppgs.numpy()

        if normalize:
            train_ppgs = standardize_sequences(train_ppgs)
            valid_ppgs = standardize_sequences(valid_ppgs)

        # 使用訓練集擬合 PCA
        pca = PCA(n_components=30)  # 假設保留10個主成分
        pca.fit(train_ppgs)
        

        if data_dir=='../train':
            self.ppgs_pca = pca.transform(train_ppgs)
            self.labels = train_set.labels
        if data_dir=='../valid':
            self.ppgs_pca = pca.transform(valid_ppgs)
            self.labels = valid_set.labels
        self.ppgs_pca = torch.tensor(self.ppgs_pca, dtype=torch.float32)
        
        
    def __len__(self):
        return len(self.ppgs_pca)  
        
    def __getitem__(self, idx):
        return  self.ppgs_pca[idx], self.labels[idx]






        



class PPGDataset(Dataset):
    def __init__(self, data_dir='../train', compare_type='Psycho', chunk_size=400, shift_step=100, 
                 aug_factor = 1, scale_mode = 'Max-Abs Scaling', pca_mode = 'Standardization', pca_n_components = 30):


        """
        # 加載保存的 .npy 文件
        rest_cleaned_chnk   = np.load(os.path.join(data_dir, "First_rest_cleaned_chunked.npy"))  # shape = [N1, chunk_size]
        psycho_cleaned_chnk = np.load(os.path.join(data_dir, "Psycho_cleaned_chunked.npy"))      # shape = [N2, chunk_size]    N1, N2沒有實際意義
        
        
        
        ae_rest   = ToolBox.extract_autoencoder_features(rest_cleaned_chnk).cpu().numpy()   # ae = autoencoder
        ae_psycho = ToolBox.extract_autoencoder_features(psycho_cleaned_chnk).cpu().numpy() 
        
        
        spectrum_rest   = ToolBox.extract_DFT_features(rest_cleaned_chnk)
        spectrum_psycho = ToolBox.extract_DFT_features(psycho_cleaned_chnk)
        """

        pre = Unqualified_PPG_preprocessor(data_dir = data_dir, compare_type = compare_type, chunk_size = chunk_size, shift_step = shift_step)

        


        
        rest_cleaned_chnk = pre.chunked_rest
        psycho_cleaned_chnk = pre.chunked_psycho

        def self_standardize_sequences(data):
            self_standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            return self_standardized_data
        self.ppg_volatility = self_standardize_sequences(pre.ppg_volatility)



        

        rest_labels   = ToolBox.get_labels(len(rest_cleaned_chnk), 'rest')
        psycho_labels = ToolBox.get_labels(len(psycho_cleaned_chnk), 'psycho')

        
        ratio_rest   = LF_RF.LF_RF_ratio(pre.raw_rest[:10000],   fs=100)
        ratio_psycho = LF_RF.LF_RF_ratio(pre.raw_psycho[:10000], fs=100)
        LF_RF_ratio = np.column_stack((ratio_rest, ratio_psycho))
        n_repeat = len(rest_cleaned_chnk) / len(LF_RF_ratio)
        assert n_repeat.is_integer(), f"n_repeat is not an integer"
        LF_RF_repeated = np.repeat(LF_RF_ratio, repeats=n_repeat, axis=0)
        self.LF_RF_ratio = np.vstack((LF_RF_repeated, LF_RF_repeated))
        

        
        self.ppgs   = np.vstack((rest_cleaned_chnk,     # size = [N1+N2, chunk_size] <class 'numpy.ndarray'>
                                 psycho_cleaned_chnk))

        self.pyscho_rest_ratio = np.array([ratio[1]/ratio[0]  for ratio in self.LF_RF_ratio])
        
            

        #self.ppgs_pca = self.PCA_calc(self.ppgs.copy(), pca_mode, pca_n_components)
        
        self.spectrums = ToolBox.rfft_with_cut_freq(self.ppgs)
        self.abs_spectrums = torch.abs(self.spectrums)
        #print('abs_spectrums : '+str(self.abs_spectrums.shape))

        """
        def self_standardize_sequences(data):
            self_standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            return self_standardized_data
            
        self.ppgs = self_standardize_sequences(self.ppgs)
        """

        
        """
        self.autoencoder_feats = np.vstack((ae_rest,     
                                            ae_psycho))
        
        self.spectrum_feats = np.vstack((spectrum_rest,     
                                         spectrum_psycho))
        """
        #self.labels = np.vstack((rest_labels,     # # size = [2M, num_of_classes] <class 'numpy.ndarray'>
        #                         psycho_labels))
        self.labels = np.concatenate((rest_labels, psycho_labels))
        
        
        self.returns = (self.ppgs[ : , 1:] / self.ppgs[ : , :-1]) - 1
        
        



        
        self.Convert_datatype_to_accustom_to_model()

    def PCA_calc(self, ppgs, pca_mode = 'Standardization', pca_n_components = 30):
        processed_ppgs = None
        if pca_mode == 'Standardization':
            scaler = StandardScaler()
            processed_ppgs = scaler.fit_transform(ppgs)
        if pca_mode == 'Raw':
            processed_ppgs = ppgs
        if pca_mode == 'Self-Standardization':
            def self_standardize_sequences(data):
                self_standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
                return self_standardized_data
            processed_ppgs = self_standardize_sequences(ppgs)

        
        pca = PCA(n_components=pca_n_components)
        ppg_signals_pca = pca.fit_transform(processed_ppgs)
        
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        print(f"保留的資訊變異比例: {explained_variance_ratio:.2%}")
        return ppg_signals_pca
            

    def normaliztion(self, aug_factor, scale_mode):
        if scale_mode == 'Max-Abs Scaling':
            return aug_factor* self.ppgs / np.max(np.abs(self.ppgs), axis=1, keepdims=True)
        if scale_mode == 'Min-Max Normalization':
            MIN, MAX = self.ppgs.min(axis=1, keepdims=True), self.ppgs.max(axis=1, keepdims=True)
            return aug_factor* (self.ppgs - MIN) / (MAX - MIN)
            
    

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
        
    def Convert_datatype_to_accustom_to_model(self):
        # 將數據類型轉換為 Tensor
        self.ppgs = torch.tensor(self.ppgs, dtype=torch.float32)
        #self.one_sided_spectrums = torch.tensor(self.one_sided_spectrums, dtype=torch.float32)

        
        #self.autoencoder_feats = torch.tensor(self.autoencoder_feats, dtype=torch.float32)
        #self.spectrum_feats = torch.tensor(self.spectrum_feats, dtype=torch.float32)
        #self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.ppgs_pca = torch.tensor(self.ppgs_pca, dtype=torch.float32)
        self.abs_spectrums = self.abs_spectrums.to(torch.float32)
        self.LF_RF_ratio = torch.tensor(self.LF_RF_ratio, dtype=torch.float32)
        self.pyscho_rest_ratio = torch.tensor(self.pyscho_rest_ratio, dtype=torch.float32)


        self.returns = torch.tensor(self.returns, dtype=torch.float32)
        self.ppg_volatility = torch.tensor(self.ppg_volatility, dtype=torch.float32)

        
        #self.spectrums = self.spectrums.to(torch.float32)
        
        """
        self.ppgs     = self.ppgs.astype(np.float32)
        #self.ppgs     = np.expand_dims(self.ppgs, axis=1)    # 因為我是用一維CNN所以要多一個維度表示channel
        self.autoencoder_feats = self.autoencoder_feats.astype(np.float32)
        self.spectrum_feats = self.spectrum_feats.astype(np.float32)
        self.labels   = self.labels.astype(np.float32)
        """


        
        
        
        
        
    def __len__(self):
        return len(self.ppgs)  # len(self.labeled_ppg) = self.labeled_ppg的row數量
        
    def __getitem__(self, idx):
        #return self.ppgs[idx], self.spectrum_feats[idx], self.autoencoder_feats[idx], self.labels[idx], self.peaks[idx], self.one_sided_spectrums[idx]
        #return self.ppgs[idx], self.peaks[idx], self.one_sided_spectrums[idx]
        return self.abs_spectrums[idx], self.ppgs[idx], self.labels[idx], self.LF_RF_ratio[idx], \
                self.pyscho_rest_ratio[idx], self.returns[idx], self.ppg_volatility[idx]
                                                            
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
#重要!!!!!!!!!!! 如果之後有DATA AUG多了NOISE，要區分是否是TRAIN還是VALID
# 不同情況需要RETURMN不同的資料











#==========================================================================================================================


class PPGDataset2(Dataset):
    def __init__(self, data_dir='../train', compare_type='Psycho', chunk_size=400, shift_step=100, 
                 aug_factor = 1, scale_mode = 'Max-Abs Scaling', pca_mode = 'Standardization', pca_n_components = 30):

        pre = Unqualified_PPG_preprocessor2(data_dir = data_dir, compare_type = compare_type, chunk_size = chunk_size, shift_step = shift_step)

        


        
        rest_cleaned_chnk = pre.chunked_rest
        psycho_cleaned_chnk = pre.chunked_psycho



        rest_labels   = ToolBox.get_labels(len(rest_cleaned_chnk), 'rest')
        psycho_labels = ToolBox.get_labels(len(psycho_cleaned_chnk), 'psycho')
    
        """
        ratio_rest   = LF_RF.LF_RF_ratio(pre.raw_rest[:10000],   fs=100)
        ratio_psycho = LF_RF.LF_RF_ratio(pre.raw_psycho[:10000], fs=100)
        LF_RF_ratio = np.column_stack((ratio_rest, ratio_psycho))
        n_repeat = len(rest_cleaned_chnk) / len(LF_RF_ratio)
        assert n_repeat.is_integer(), f"n_repeat is not an integer"
        LF_RF_repeated = np.repeat(LF_RF_ratio, repeats=n_repeat, axis=0)
        self.LF_RF_ratio = np.vstack((LF_RF_repeated, LF_RF_repeated))
        """

        
        self.ppgs   = np.vstack((rest_cleaned_chnk,     # size = [N1+N2, chunk_size] <class 'numpy.ndarray'>
                                 psycho_cleaned_chnk))

        #self.pyscho_rest_ratio = np.array([ratio[1]/ratio[0]  for ratio in self.LF_RF_ratio])
        
            
        """
        ppgs_pca = self.PCA_calc(np.vstack((pre.raw_rest, pre.raw_psycho)), 
                                 pca_mode, pca_n_components)
        repeated_times = len(pre.chunked_rest) / len(pre.raw_rest)
        self.ppgs_pca = np.repeat(ppgs_pca, repeated_times, axis=0)
        """
        
        #self.spectrums = ToolBox.rfft_with_cut_freq(self.ppgs)
        #self.abs_spectrums = torch.abs(self.spectrums)
        #print('abs_spectrums : '+str(self.abs_spectrums.shape))

        """
        def self_standardize_sequences(data):
            self_standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            return self_standardized_data
            
        self.ppgs = self_standardize_sequences(self.ppgs)
        """

        
        """
        self.autoencoder_feats = np.vstack((ae_rest,     
                                            ae_psycho))
        
        self.spectrum_feats = np.vstack((spectrum_rest,     
                                         spectrum_psycho))
        """
        #self.labels = np.vstack((rest_labels,     # # size = [2M, num_of_classes] <class 'numpy.ndarray'>
        #                         psycho_labels))
        self.labels = np.concatenate((rest_labels, psycho_labels))
        
        
        #self.returns = (self.ppgs[ : , 1:] / self.ppgs[ : , :-1]) - 1
        #self.curvatures = (self.returns[ : , 1:] / self.returns[ : , :-1]) - 1
        #self.ppg_volatility = ToolBox.get_ppg_volatility(self.returns, window_size = 100)
        print('VOLO')
        



        
        self.Convert_datatype_to_accustom_to_model()

    def PCA_calc(self, ppgs, pca_mode = 'Standardization', pca_n_components = 30):
        processed_ppgs = None
        if pca_mode == 'Standardization':
            scaler = StandardScaler()
            processed_ppgs = scaler.fit_transform(ppgs)
        if pca_mode == 'Raw':
            processed_ppgs = ppgs
        if pca_mode == 'Self-Standardization':
            def self_standardize_sequences(data):
                self_standardized_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
                return self_standardized_data
            processed_ppgs = self_standardize_sequences(ppgs)

        
        pca = PCA(n_components=pca_n_components)
        ppg_signals_pca = pca.fit_transform(processed_ppgs)
        
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        print(f"保留的資訊變異比例: {explained_variance_ratio:.2%}")
        return ppg_signals_pca
            

    def normaliztion(self, aug_factor, scale_mode):
        if scale_mode == 'Max-Abs Scaling':
            return aug_factor* self.ppgs / np.max(np.abs(self.ppgs), axis=1, keepdims=True)
        if scale_mode == 'Min-Max Normalization':
            MIN, MAX = self.ppgs.min(axis=1, keepdims=True), self.ppgs.max(axis=1, keepdims=True)
            return aug_factor* (self.ppgs - MIN) / (MAX - MIN)
            
    

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
        
    def Convert_datatype_to_accustom_to_model(self):
        # 將數據類型轉換為 Tensor
        self.ppgs = torch.tensor(self.ppgs, dtype=torch.float32)
        #self.one_sided_spectrums = torch.tensor(self.one_sided_spectrums, dtype=torch.float32)

        
        #self.autoencoder_feats = torch.tensor(self.autoencoder_feats, dtype=torch.float32)
        #self.spectrum_feats = torch.tensor(self.spectrum_feats, dtype=torch.float32)
        #self.ppgs_pca = torch.tensor(self.ppgs_pca, dtype=torch.float32)
        #self.abs_spectrums = self.abs_spectrums.to(torch.float32)
        #self.LF_RF_ratio = torch.tensor(self.LF_RF_ratio, dtype=torch.float32)
        #self.pyscho_rest_ratio = torch.tensor(self.pyscho_rest_ratio, dtype=torch.float32)


        #self.returns = torch.tensor(self.returns, dtype=torch.float32)
        #self.ppg_volatility = torch.tensor(self.ppg_volatility, dtype=torch.float32)
        #self.curvatures = torch.tensor(self.curvatures, dtype=torch.float32)
        
        #self.spectrums = self.spectrums.to(torch.float32)
        
        """
        self.ppgs     = self.ppgs.astype(np.float32)
        #self.ppgs     = np.expand_dims(self.ppgs, axis=1)    # 因為我是用一維CNN所以要多一個維度表示channel
        self.autoencoder_feats = self.autoencoder_feats.astype(np.float32)
        self.spectrum_feats = self.spectrum_feats.astype(np.float32)
        self.labels   = self.labels.astype(np.float32)
        """


        
        
        
        
        
    def __len__(self):
        return len(self.ppgs)  # len(self.labeled_ppg) = self.labeled_ppg的row數量
        
    def __getitem__(self, idx):
        #return self.ppgs[idx], self.spectrum_feats[idx], self.autoencoder_feats[idx], self.labels[idx], self.peaks[idx], self.one_sided_spectrums[idx]
        #return self.ppgs[idx], self.peaks[idx], self.one_sided_spectrums[idx]
        #return self.abs_spectrums[idx], self.ppgs[idx], self.labels[idx], self.LF_RF_ratio[idx], \
        #        self.pyscho_rest_ratio[idx], self.returns[idx], self.ppg_volatility[idx], self.curvatures[idx]
        return self.ppgs[idx], self.labels[idx]
                                                            
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
#重要!!!!!!!!!!! 如果之後有DATA AUG多了NOISE，要區分是否是TRAIN還是VALID
# 不同情況需要RETURMN不同的資料