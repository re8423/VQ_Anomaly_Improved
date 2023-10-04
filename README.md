# VQ_Anomaly_Improved

This Repo contains code from my 3rd year dissertation: ["Improved Density-Estimation and Restoration based Vector-Quantized Anomaly Detection"](https://github.com/re8423/VQ_Anomaly_Improved/blob/ba4f2fd8d63ff37f4d69939bf843be97c785da53/Improved_Density-Estimation_and_Restoration_based_Vector-Quantized_Anomaly_Detection.pdf) (83/100)




* VQ_Model
  >
  >This folder contains the VQ_VAE and VQ_GAN classes.
  >

* dfkde.py, ganomaly.py 
  >
  >Contains code to run DFKDE and Ganomaly from the Anomalib Library.
  >

* mingpt.py, pixelsnail.py 
  >
  >Contains code for the mingpt architecure and pixelsnail architecure.
  >

Training Loops are in their respective _train files.

Here is a simple Demo of the Light pixel-wise detection method used in conjunction with a VQ-VAE and Transformer Combo:
![Gifdemo](https://github.com/re8423/VQ_Anomaly_Improved/blob/main/Images/light_wise_vae_trans_anom(1).gif)
