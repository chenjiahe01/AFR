from abc import ABC,abstractclassmethod
import os
import numpy as np
from secml.array import CArray
from secml_malware.models import CClassifierEnd2EndMalware, MalConv
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi, CEnd2EndWrapperPhi
from dotenv import dotenv_values
class AFR(ABC):

    def __init__(self) -> None:
        # self.config = dotenv_values(".env")
        self.model = None
        # self.model
        self.afList = []
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.threshold = None
        self.dense_1_weight = None
        self.dense_2_weight = None
        self.afd = None

    @abstractclassmethod
    def load_dnn_model(self, path) -> None:
        """
        load dnn-based malware detector model(Malconv, Malconv2) from path. 
        """
        pass

    def detectFile(self, samplePath):
        """
        detect a sample PE file using dnn-based malware-detector. Return label, confidence and features activation.
        """
        with open(samplePath,'rb') as handle:
            code = handle.read()
        code = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
        padded_x = CArray.zeros((code.shape[0], self.model.get_input_max_length())) + self.model.get_embedding_value()
        for i in range(code.shape[0]):
            x_i = code[i, :]
            length = min(x_i.shape[-1], self.model.get_input_max_length())
            padded_x[i, :length] = x_i[0, :length] + self.model.get_is_shifting_values()
        y_pred,conf= self.model.predict(padded_x, return_decision_function=True)

        y_pred = y_pred.item()
        score = conf[0][0, 1].item()
        fv_adv = conf[1]
        return y_pred,score,fv_adv

    def detectPath(self, path):
        """
        detect all sample PE files under the path using dnn-based malware-detector. Return label, confidence and features activation.
        """
        benign = 0 
        malicious = 0
        files = os.listdir(path)
        for file in files:
            filepath = path + file
            label, score, fv_adv = self.detectFile(filepath)
            if label == 0:
                benign += 1
            else:
                malicious += 1
        return benign, malicious
    @abstractclassmethod
    def getAFD(self):
        """
        get Adversarial-Feature-Distribution(AFR) by comparing feature activation between adversarial-original sample pairs.
        """
        pass
    
    def setAFR(self, alpha, beta, gamma ,ori_path, adv_path):
        """
        set params of AFR.
        """
        afd,_,_,_ = self.getAFD(ori_path,adv_path)
        afList = []
        for i in range(len(afd)):
            if afd[i] > alpha:
                afList.append(i)
        self.afList = afList
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma    
        self.afd = afd

    def detectFile_guarded(self, samplePath):
        """
        detect a sample PE file using dnn-based malware-detector repaired by AFR. Return label, confidence. 
        """
        with open(samplePath,'rb') as handle:
            code = handle.read()
        code = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
        padded_x = CArray.zeros((code.shape[0], self.model.get_input_max_length())) + self.model.get_embedding_value()
        for i in range(code.shape[0]):
            x_i = code[i, :]
            length = min(x_i.shape[-1], self.model.get_input_max_length())
            padded_x[i, :length] = x_i[0, :length] + self.model.get_is_shifting_values()
        y_pred,conf= self.model.guarded_embedding_predict(padded_x, self.afList, self.beta, self.gamma, self.dense_2_weight)
        y_pred = y_pred.item()
        score = conf[0][0, 1].item()
        fv_adv = conf[1]
        return y_pred,score,fv_adv
    
    def detectPath_guarded(self, path):
        """
        detect all sample PE files under the path using dnn-based malware-detector repaired by AFR. Return label, confidence and features activation.
        """
        benign = 0 
        malicious = 0
        files = os.listdir(path)
        for file in files:
            filepath = path + file
            label, score, fv_adv = self.detectFile_guarded(filepath)
            if label == 0:
                benign += 1
            else:
                malicious += 1
        return benign, malicious
    

    



