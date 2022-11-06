# projet feux de forêt



## Introduction:
Forest fires are a huge concern in the world. With global warming, fires are even more frequent causing a chain reaction (forest fire->CO2 emission->global warming->more forst fire).  
This project consists of Trying to detect forest fire through sound, which could improve fire detection and prevention.
## Members:
##### Élie Caratgé
##### Nizar El Ghazal
##### Basile Heurtault
##### Oussama Kharouiche
##### Wais Lefevre
## Files structure :
### Datasets : contains all the datasets used in AI training 
##### the datasets are not direclty included because of size issues, you can download them fro the link : https://centralesupelec-my.sharepoint.com/:u:/g/personal/nizar_elghazal_student-cs_fr/ESQtsAqx825Dsh46uReUzwkBwjntqABfdl8QFvmUBRAE8g?e=jISz5l

- Brut and Untouched : were used temporary folders to store the the outputs of division.py  
- Samples : contains the dataset which is divided into fire sounds and no fire sounds (other)  
### Presentation+Rapport : pretty much sef explanatory
### Files : contains the other files used in this project :
##### Audio : folder containing the audio files used to test the AI, and contains the recordings made with the raspberry  
##### Biblio : all the documetns related to the bibliography  
##### Program : Contains all codes and trained AI  
###### Trained neural network models :  
- model_0 : first CNN model, dataset doesn't contains any noise, STFT focuses on time ddefinition.  
- model_1 : same dataset as first, STFT focuses on frequency ddefinition.  
- model_noise_0 : augmented the datasets with noise, TFT focuses on time ddefinition.  
- model_noise_1 : same dataset as the previous one, STFT focuses on frequency ddefinition.  
- lite models : model_noise_0 and model_noise_1, but converted to support tensorflow lite   
###### Modules : contains the scripts to train the CNN and the supporting functions used to classify sounds with the CNN
###### division.py : scripts used to divide and convert audio files, and to mix 2 audio files 
###### liteconvert.py : convert a tensorflow 2 model into a tensorflow lite model
###### plot_data.py : plots the spectrogram of audio files
###### program.py : calls the CNN on an audio file to classify it
###### record.py : script used on the raspberry to record audio files and save them in Audio folder
##### Results : containts waveforms and spectrograms of some audio files.


### IMPORTANT! : To avoid any problems running any of the scripts, please set the main folder (the one containing README.md) as your current working directory 
