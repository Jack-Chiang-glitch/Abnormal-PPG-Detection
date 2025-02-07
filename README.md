# Abnormal-PPG-Detection (Provide both Chinese and English versions of README.md.)


Data Preprocessing : <br>
對於每一條訊號(5分鐘，sampling rate=100)，先clip距離整條訊號平均的正負兩個標準差<br>
μ = mean(訊號) <br>
σ = std(訊號)<br>
Clip(μ-2σ, μ+2σ)去除掉，避免過大的振幅影響到接下來濾波完的資料品質<br>
利用Butterworth濾波器(order=2)，濾掉0.5HZ~5HZ的雜訊，<br>



Data Augmentation :<br>
模型的輸入是每10秒切成一段，但是由於受試者樣本數非常的小，所以適當做資料擴增可以避免過度擬合，還可以讓模型變的穩健。<br>
將每一段訊號向右平移1秒，如此一來，就獲得了10備的資料量。<br>

LSTM-Transformer based model :<br>
模型架構 : 
![image](https://github.com/user-attachments/assets/5dc881df-adb1-407d-953b-6868ad30ded9)

LSTM-Transformer對於抓取時間序列前後的關聯非常有效，在分類正常訊號以及有壓力的訊號上，取得相當高的準確率。<br>
Train set acc: 0.999 Test set acc : 0.88<br>

以上是在Time Domain上分析<br>


但是訊號也可以用Frequency Domain分析 :<br>
# (1)傅立葉變換 : <br>
成效非常不好，因為各個訊號振幅差異過大，而且單純做傅立葉有一個嚴重的缺陷，<br>
做傅立葉轉換是看全局的資訊，如果中間有雜訊或是飄移，都會嚴重影響模型的輸入。<br>
但是LSTM-Transformer在Time domain上萃取特徵的時候是用CNN，而CNN是捕捉局部的性質，<br>
只需要顧及鄰近的區域，所以飄移對於CNN來說可以忽略，變成假設訊號是沒有drift，因此效果很好。<br>

而且傅立葉變換完全沒有時間資訊，所以不知道個個頻率出現的位置，因此需要(2)<br>
# (2)短時距傅立葉變換 : <br>
利用小長度的窗口做傅立葉轉換，畫出spectrogram，但是短時距傅立葉變換有一個問題，如果選用窗口過小，捕捉不到低頻，<br>
選用長度過大，高頻資訊會消失在spectrum當中，因此需要用到(3)小波分析<br>
Window = 20(0.2 sec)<br>
![image](https://github.com/user-attachments/assets/c15ac166-ae04-44e0-9caa-d8a0e56e425f)

Window = 500(5 sec)<br>
![image](https://github.com/user-attachments/assets/af18fc98-fc1c-421a-b92f-796f85bd3d4b)



# (3)小波分析:<br>
可以選用母小波以及控制中心頻率還有scale，捕捉不同尺度的頻率，畫出scalogram<br>
丟入LSTM-transformer訓練，也獲得好的成效<br>
可以清晰看出頻率在時間出現的位置，但是並非毫無代價<br>
帶寬增加（Bandwidth ⬆）<br>
高頻率解析度，但低時間解析度<br>

帶寬減少（Bandwidth ⬇）<br>
高時間解析度，但低頻率解析度<br>

cmor4.0-3.0 <br>
(complex Morlet, bandwidt=4, fc=3)<br>
Scales = 40~1000 (300 points with log scale)<br><br>
![image](https://github.com/user-attachments/assets/d6c5ee8e-aa2b-4f27-94cc-161c927e5de4)




# Data Preprocessing:<br>
For each signal (5 minutes, sampling rate = 100 Hz), we first clip values that exceed two standard deviations from the mean to remove extreme amplitudes that may affect the quality of the filtered data.<br>

Let:<br>
μ = mean(signal)<br>
σ = standard deviation(signal)<br>
Clipping range: [μ−2σ,μ+2σ]<br>
This process prevents large amplitude outliers from distorting the subsequent filtered signals.<br>

We then apply a Butterworth filter (order = 2) to remove noise in the 0.5 Hz – 5 Hz range.<br>

Data Augmentation:<br>
The model input consists of 10-second signal segments. However, due to the limited number of subjects, applying data augmentation can help prevent overfitting and improve the model’s robustness.<br>

To achieve this, each signal segment is shifted 1 second to the right, effectively increasing the dataset size by a factor of 10.<br>

# LSTM-Transformer based model :<br>
model architecture : <br>
![image](https://github.com/user-attachments/assets/5dc881df-adb1-407d-953b-6868ad30ded9)

LSTM-Transformer for Time Series Analysis<br>
LSTM-Transformer is highly effective in capturing temporal dependencies in time series data. It achieves high accuracy in classifying normal and stress signals:<br>

Train set accuracy: 0.999<br>
Test set accuracy: 0.88<br>
The above analysis was conducted in the Time Domain.<br>

Frequency Domain Analysis:<br>
# (1) Fourier Transform<br>
The performance of Fourier Transform was very poor due to the large amplitude variations across signals. Additionally, Fourier Transform has a major drawback—it captures only global information. If noise or drift exists in the middle of the signal, it can significantly affect the model’s input.<br>

However, in the Time Domain, LSTM-Transformer extracts features using CNN, which captures local properties.<br>

CNN only needs to consider neighboring regions, making it robust to drift.<br>
This effectively assumes the signal has no drift, leading to better performance.<br>
Moreover, Fourier Transform lacks time information, meaning it cannot determine when specific frequencies appear. This leads to the need for (2) Short-Time Fourier Transform (STFT).<br>

# (2) Short-Time Fourier Transform (STFT)<br>
STFT applies Fourier Transform on small time windows, generating a spectrogram. However, STFT has an inherent trade-off:<br>

If the window size is too small, it fails to capture low-frequency components.<br>
If the window size is too large, high-frequency information is lost in the spectrum.<br>
To address this limitation, (3) Wavelet Transform is needed.<br>

Chosen window size: 20 samples (0.2 sec)<br>
![image](https://github.com/user-attachments/assets/c15ac166-ae04-44e0-9caa-d8a0e56e425f)
Chosen window size: 500 samples (5 sec)<br>
![image](https://github.com/user-attachments/assets/af18fc98-fc1c-421a-b92f-796f85bd3d4b)


# (3) Wavelet Analysis:<br>
Wavelet Transform allows the selection of mother wavelets, as well as the control of central frequency and scale, enabling the capture of frequencies at different scales and generating a scalogram.<br>

When fed into the LSTM-Transformer, this approach also achieves good performance.<br>
Wavelet analysis provides clear visibility of when specific frequencies appear over time, but this advantage comes with trade-offs:<br>

Increased bandwidth (Bandwidth ⬆)<br>

Higher frequency resolution, but lower time resolution<br>
Decreased bandwidth (Bandwidth ⬇)<br>

Higher time resolution, but lower frequency resolution<br>
The parameters used:<br>

Mother wavelet: cmor4.0-3.0 (Complex Morlet, bandwidth = 4, central frequency = 3)<br>
Scales: 40~1000 (300 points on a logarithmic scale)<br>










![image](https://github.com/user-attachments/assets/d6c5ee8e-aa2b-4f27-94cc-161c927e5de4)








