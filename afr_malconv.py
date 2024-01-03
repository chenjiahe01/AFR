from afr import AFR
import numpy as np
import os
from secml_malware.models import  MalConv
from secml_malware.models.c_classifier_end2end_malware_afr import CClassifierEnd2EndMalware
class AFR_MalConv(AFR):
    def __init__(self) -> None:
        super().__init__()
    def load_dnn_model(self,path=None):
        self.model = CClassifierEnd2EndMalware(model=MalConv())
        self.model.load_pretrained_model(path)
        self.dense_1_weight = self.model.model.dense_1.weight
        self.dense_2_weight = self.model.model.dense_2.weight[0]
    def getAFD(self, original_path, adversarial_path):
        """
        get Adversarial-Feature-Distribution(AFR) by comparing feature activation between adversarial-original sample pairs.
        """
        freq = np.zeros(128)
        freq=freq.astype(int)
        ave = np.zeros(128)
        add = np.zeros(128).astype(int)
        dec = np.zeros(128).astype(int)
        adv_files = os.listdir(adversarial_path)
        j = 0
        for adv in adv_files:
            if j == 1000:
                break
            adv_pred, adv_score, adv_fv = self.detectFile(adversarial_path+adv)
            ori = adv.split('_')[0].split('turns')[0].split('.')[0]
            ori_pred, ori_score, ori_fv = self.detectFile(original_path+ori)
            fv_diff = adv_fv - ori_fv
            for i in range(128):
                ave[i] = (j*ave[i]+fv_diff[0][i])/(j+1)
                if fv_diff[0][i]!=0:
                    freq[i] += 1
                if fv_diff[0][i] >0:
                    add[i] += 1
                if fv_diff[0][i] <0:
                    dec[i] += 1
            j = j+1
        return freq/1000,add/1000,dec/1000,ave
