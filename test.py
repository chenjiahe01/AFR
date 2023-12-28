import argparse
import os
import numpy as np
from secml.array import CArray
import torch
import csv
import time
from utils import create_wrapper_for_global_target
from secml_malware.models import CClassifierEnd2EndMalware, MalConv
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi, CEnd2EndWrapperPhi
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import dotenv_values
config = dotenv_values('.env')
weight = pd.read_excel('./data/fc2_weight.xls',sheet_name=0)
weight = weight["weight"].tolist()
afLists = [[12,18,36,95,102,124],[12,18,36,95,102,113,124],[12,18,36,83,95,102,113,124],[6,12,18,36,83,95,102,113,124],[6,12,18,36,83,95,102,108,113,124]]

clf = CClassifierEnd2EndMalware(model=MalConv())
clf.load_pretrained_model()
net = CEnd2EndWrapperPhi(clf)

clf1 = CClassifierEnd2EndMalware(model=MalConv())
clf1.load_pretrained_model("/home/cjh/malconv_retrain/save/malconv/max-acc-adv-gs.pth")
net1 = CEnd2EndWrapperPhi(clf1)

clf2 = CClassifierEnd2EndMalware(model=MalConv())
clf2.load_pretrained_model("/home/cjh/malconv_retrain/save/malconv/max-acc-adv-mab.pth")
net2 = CEnd2EndWrapperPhi(clf2)

clf3 = CClassifierEnd2EndMalware(model=MalConv())
clf3.load_pretrained_model("/home/cjh/malconv_retrain/save/malconv/max-acc-adv-kreuk.pth")
net3 = CEnd2EndWrapperPhi(clf3)

def detect_file(clf,path):
    with open(path,'rb') as handle:
        code = handle.read()
    code = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    padded_x = CArray.zeros((code.shape[0], clf.get_input_max_length())) + clf.get_embedding_value()
    for i in range(code.shape[0]):
        x_i = code[i, :]
        length = min(x_i.shape[-1], clf.get_input_max_length())
        padded_x[i, :length] = x_i[0, :length] + clf.get_is_shifting_values()
    y_pred,conf= clf.predict(padded_x, return_decision_function=True)
    # print("conf:",conf)
    y_pred = y_pred.item()
    score = conf[0][0, 1].item()
    fv_adv = conf[1]
    return y_pred,score,fv_adv

def compareWeight(m1,m2):
    layerWeight1 = m1.model.dense_2.weight
    layerWeight2 = m2.model.dense_2.weight
    diff = layerWeight2 - layerWeight1
    for i in range(128):
        if diff[i] != 0:
            print("i: "+str(i)+" diff: "+str(diff[i].detach())+' '+str(diff[i].detach()/layerWeight1[i].detach()))
    # print(diff)
def compareAdvOri(adv_path,ori_path):
    freq = np.zeros(128)
    freq=freq.astype(int)
    ave = np.zeros(128)
    add = np.zeros(128).astype(int)
    dec = np.zeros(128).astype(int)
    # layerWeight = getWeight(clf)
    layerWeight = clf.model.dense_2.weight
    adv_files = os.listdir(adv_path)
    j = 0
    for adv in adv_files:
        if j == 1000:
            break
        adv_pred, adv_score, adv_fv = detect_file(clf,adv_path+adv)
        # print(adv_fv)
        ori = adv.split('_')[0].split('turns')[0].split('.')[0]
        ori_pred, ori_score, ori_fv = detect_file(clf,ori_path+ori)
        # print(ori_fv)
        fv_diff = adv_fv - ori_fv
        # print(fv_diff)
        for i in range(128):
            # print(fv_diff[0][i])
            ave[i] = (j*ave[i]+fv_diff[0][i])/(j+1)
            if fv_diff[0][i]!=0:
                freq[i] += 1
            if fv_diff[0][i] >0:
                add[i] += 1
            if fv_diff[0][i] <0:
                dec[i] += 1
        j = j+1
    # print("id           freq            weight          add         dec         ave")
    # x = []
    # for i in range(128):
    #     x.append(i)
    # for i in range(128):
    #     print("{}           {}          {}          {}          {}          {}".format(i,freq[i],layerWeight[0][i],add[i],dec[i],ave[i]))
    return freq/1000,layerWeight[0],add/1000,dec/1000,ave
def detectPath(clf, path):
    start = time.time()
    benign = 0
    files = os.listdir(path)
    total = len(files)
    activ = np.zeros(128)
    j= 0
    for file in files:
        if j == 1000:
            break
        pred, score, fv = detect_file(clf, path+file)
        for i in range(128):
            if fv[0][i]>0:
                activ[i]+=1
        print("result: {},score: {}".format(pred,score))
        if pred == 0:
            # os.system("rm "+path+file)
            benign += 1
        j+=1

    end = time.time()
    consum = end - start
    print("time consum: ",consum)
    evasive_rate = benign/total
    detect_rate = (total-benign)/total
    print("malicious num: ",total-benign)
    print("benign num: ",benign)
    print("detect rate: ",detect_rate)
    print("evasive rate: ",evasive_rate)
def getWeight(model):
    weight1 = model.model.dense_1.weight       #全连接层1的权重矩阵
    weight2 = model.model.dense_2.weight[0]     #全连接层2的权重矩阵
    cam_weight = np.array([0]*128)                      #逻辑上将两层全连接层权重矩阵合并为一个权重向量cam_weight
    cam_weight = torch.from_numpy(cam_weight)
    cam_weight = cam_weight.float().cuda()
    for i in range(0,128) :
        for row in weight1 :
            cam_weight[i] += row[i]*weight2[i]
    # for i in range(0,128):
    #     print("id: {}       weight: {}".format(i,cam_weight[i]))
    return [cam_weight], weight2

def getFCweight(model):
    x = []
    for i in range(128):
        x.append(i)
    with open('./data/fc2_weight.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature id','weight','freq_kreuk','freq_gs','freq_mab'])
        weight = model.model.dense_2.weight[0].cpu().detach()
        allstart = time.time()
        ks = time.time()
        freq_kreuk,layerWeight_kreuk,add_kreuk,dec_kreuk,ave_kreuk = compareAdvOri(config.get("kreuk_samples"),'../pe3test/') 
        print("kreuk use "+str(time.time()-ks))
        freq_gs,layerWeight_gs,add_gs,dec_gs,ave_gs = compareAdvOri(config.get("gs_samples"),'../pe3test/') 
        freq_mab,layerWeight_mab,add_mab,dec_mab,ave_mab = compareAdvOri(config.get("mab_samples"),'../pe3test/') 
        allend = time.time()
        print("all sonsum: "+str(allend-allstart))
        for i in x :
            writer.writerow([i,float(weight[i]),freq_kreuk[i],freq_gs[i],freq_mab[i]])
        fig, axs = plt.subplots(3, 1,figsize=(30, 20))
        axs[0].bar(x, freq_kreuk)  # 画柱状图  
        axs[0].set_ylim([0,1])
        axs[0].set_xlabel('kreuk samples feature id',fontsize=30)  # 设置横轴标签  S
        axs[0].set_ylabel('modified frequence',fontsize=30)  # 设置纵轴标签  
        axs[0].tick_params(axis='both', which='major', labelsize=30)

        axs[1].bar(x, freq_gs)  # 画柱状图  
        axs[1].set_ylim([0,1])
        axs[1].set_xlabel('gs samples feature id',fontsize=30)  # 设置横轴标签  
        axs[1].set_ylabel('modified frequence',fontsize=30)  # 设置纵轴标签  
        axs[1].tick_params(axis='both', which='major', labelsize=30)

        axs[2].bar(x, freq_mab)  # 画柱状图  
        axs[2].set_ylim([0,1])
        axs[2].set_xlabel('mab samples feature id',fontsize=30)  # 设置横轴标签  
        axs[2].set_ylabel('modified frequence',fontsize=30)  # 设置纵轴标签  
        axs[2].tick_params(axis='both', which='major', labelsize=30)
        plt.savefig('afd.pdf',bbox_inches='tight',pad_inches=0.0,dpi = 500)

def getWeightFig():
    _,weight = getWeight(clf)
    weight = weight.cpu().detach().numpy()
    weight_abs = [abs(i) for i in weight]
    color = []
    for i in weight:
        if i>0:
            color.append('b')
        if i<0:
            color.append('orange')
    x = []
    for i in range(128):
        x.append(i)
    fig, axs = plt.subplots(1, 1,figsize=(30, 20))
    axs.bar(x,weight,color = color)
    axs.set_xlabel('feature ID',fontsize=20)
    axs.set_ylabel('weight',fontsize=20)
    positive_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', label='Positive')
    negative_patch = plt.Rectangle((0, 0), 1, 1, fc='orange', label='Negative')
    axs.tick_params(axis='both', which='major', labelsize=20)
    axs.legend(handles=[positive_patch,negative_patch], loc='upper right',fontsize=20)
    plt.savefig('weight.pdf',bbox_inches='tight',pad_inches=0.0,dpi = 500)
    plt.show()

def guarded_detect_file(path, afList ,beta , gamma , weight):
    with open(path,'rb') as handle:
        code = handle.read()
    code = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    padded_x = CArray.zeros((code.shape[0], clf.get_input_max_length())) + clf.get_embedding_value()
    for i in range(code.shape[0]):
        x_i = code[i, :]
        length = min(x_i.shape[-1], clf.get_input_max_length())
        padded_x[i, :length] = x_i[0, :length] + clf.get_is_shifting_values()
    y_pred,conf= clf.guarded_embedding_predict(padded_x, afList , beta , gamma , weight)
    y_pred = y_pred.item()
    score = conf[0][0, 1].item()
    fv_adv = conf[1]
    return y_pred,score,fv_adv


def guarded_detectPath(path , alList ,beta , gamma , weight):
    benign = 0
    files = os.listdir(path)
    total = len(files)
    activ = np.zeros(128)
    j= 0
    for file in files:
        if j == 100:
            break
        pred, score, fv = guarded_detect_file(path+file, alList ,beta , gamma , weight)
        for i in range(128):
            if fv[0][i]>0:
                activ[i]+=1
        if pred == 0:
            benign += 1
        j+=1
    # print("time consum: ",consum)
    evasive_rate = benign/1000
    detect_rate = 1-evasive_rate
    # print("malicious num: ",total-benign)
    # print("benign num: ",benign)
    # print("detect rate: ",detect_rate)
    # print("evasive rate: ",evasive_rate)
    return evasive_rate,detect_rate
def findParaBeta(weight,afLists):
    gamma = 1
    j=0
    alpha = [0.9,0.8,0.7,0.6,0.5]
    for afList in afLists:
        print("adverarial feature ids:",afList)
        al = alpha[j]
        print("alpha is:",al)
        j = j+1
        al = str(al)
        with open('./data/find_beta/'+'alpha='+al+'_find_beta.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['alpha','beta','dr_kreuk_samples','dr_gs_samples','dr_mab_samples','dr_benign_samples','dr_mal_samples'])
        drs_kreuk = []
        drs_gs = []
        drs_mab = []
        drs_mal = []
        drs_benign = []
        for beta in [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1]:
            
            print("beta is:",beta)
            print("-------------------------------------------------------------------------")
            print("processing kreuk samples")
            _,dr_kreuk_samples = guarded_detectPath(config.get('kreuk_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_kreuk_samples)
            print("-------------------------------------------------------------------------")
            print("processing gs asmples")
            _,dr_gs_samples = guarded_detectPath(config.get('gs_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_gs_samples)
            print("-------------------------------------------------------------------------")
            print("processing mab asmples")
            _,dr_mab_samples = guarded_detectPath(config.get('mab_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_mab_samples)
            print("-------------------------------------------------------------------------")
            print("processing malware asmples")
            _,dr_mal = guarded_detectPath(config.get('test_path'),afList,beta,gamma,weight)
            print("detection rate:",dr_mal)
            print("-------------------------------------------------------------------------")
            print("processing benign asmples")
            _,dr_benign = guarded_detectPath(config.get('benign_samples'),afList,beta,gamma,weight)
            
            dr_benign = 1-dr_benign
            print("detection rate:",dr_benign)
            drs_kreuk.append(dr_kreuk_samples)
            drs_gs.append(dr_gs_samples)
            drs_mab.append(dr_mab_samples)
            drs_mal.append(dr_mal)
            drs_benign.append(dr_benign)
            with open('./data/find_beta/'+'alpha='+al+'_find_beta.csv','a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([al,str(beta),str(dr_kreuk_samples),str(dr_gs_samples),str(dr_mab_samples),str(dr_benign),str(dr_mal)])
        fig, ax = plt.subplots(1, 1,figsize=(20, 20))
        ax.set_xlim([1,-0.1])
        ax.plot([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1], drs_kreuk, linewidth= 2, color="red", label='Kreuk'  )
        ax.plot([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1], drs_gs, linewidth= 2,color="black", label='Gamma Sections')
        ax.plot([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1], drs_mab, linewidth= 2,color="blue", label='MAB')
        ax.plot([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1], drs_mal, linewidth= 2,color="green", label='Malware')
        ax.plot([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1], drs_benign, linewidth= 2,color="orange", label='Benign')
        ax.set_xlabel('β',fontsize=20)  # 设置横轴标签  
        ax.set_ylabel('detection rate',fontsize=20)  # 设置纵轴标签  
        ax.tick_params(axis='both', which='major', labelsize=20)
        legend=ax.legend(fontsize=20)
        plt.savefig('./data/find_beta/weight_alpha='+al+'.pdf',bbox_inches='tight',pad_inches=0.0,dpi = 500)

def findParaGamma(weight,afLists):
    beta = 1
    j=0
    alpha = [0.9,0.8,0.7,0.6,0.5]
    for afList in afLists:
        print("adverarial feature ids:",afList)
        al = alpha[j]
        print("alpha is:",al)
        j = j+1
        al = str(al)
        with open('./data/find_gamma/'+'alpha='+al+'_find_gamma.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['alpha','gamma','dr_kreuk_samples','dr_gs_samples','dr_mab_samples','dr_benign_samples','dr_mal_samples'])
        drs_kreuk = []
        drs_gs = []
        drs_mab = []
        drs_mal = []
        drs_benign = []
        for gamma in [1,10,20,30,40,50,60,70,80,90,100]:
            
            print("gamma is:",gamma)
            print("-------------------------------------------------------------------------")
            print("processing kreuk samples")
            _,dr_kreuk_samples = guarded_detectPath(config.get('kreuk_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_kreuk_samples)
            print("-------------------------------------------------------------------------")
            print("processing gs asmples")
            _,dr_gs_samples = guarded_detectPath(config.get('gs_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_gs_samples)
            print("-------------------------------------------------------------------------")
            print("processing mab asmples")
            _,dr_mab_samples = guarded_detectPath(config.get('mab_samples'),afList,beta,gamma,weight)
            print("detection rate:",dr_mab_samples)
            print("-------------------------------------------------------------------------")
            print("processing malware asmples")
            _,dr_mal = guarded_detectPath(config.get('test_path'),afList,beta,gamma,weight)
            print("detection rate:",dr_mal)
            print("-------------------------------------------------------------------------")
            print("processing benign asmples")
            _,dr_benign = guarded_detectPath(config.get('benign_samples'),afList,beta,gamma,weight)
            
            dr_benign = 1-dr_benign
            print("detection rate:",dr_benign)
            drs_kreuk.append(dr_kreuk_samples)
            drs_gs.append(dr_gs_samples)
            drs_mab.append(dr_mab_samples)
            drs_mal.append(dr_mal)
            drs_benign.append(dr_benign)
            with open('./data/find_gamma/'+'alpha='+al+'_find_gamma.csv','a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([al,str(gamma),str(dr_kreuk_samples),str(dr_gs_samples),str(dr_mab_samples),str(dr_benign),str(dr_mal)])
        fig, ax = plt.subplots(1, 1,figsize=(20, 20))
        ax.set_xlim([1,100])
        ax.plot([1,10,20,30,40,50,60,70,80,90,100], drs_kreuk, linewidth= 3, color="red", label='Kreuk'  )
        ax.plot([1,10,20,30,40,50,60,70,80,90,100], drs_gs, linewidth= 3,color="black", label='Gamma Sections')
        ax.plot([1,10,20,30,40,50,60,70,80,90,100], drs_mab, linewidth= 3,color="blue", label='MAB')
        ax.plot([1,10,20,30,40,50,60,70,80,90,100], drs_mal, linewidth= 3,color="green", label='Malware')
        ax.plot([1,10,20,30,40,50,60,70,80,90,100], drs_benign, linewidth= 3,color="orange", label='Benign')
        ax.set_xlabel('γ',fontsize=20)  # 设置横轴标签  
        ax.set_ylabel('detection rate',fontsize=20)  # 设置纵轴标签  
        ax.tick_params(axis='both', which='major', labelsize=20)
        legend=ax.legend(fontsize=20)
        plt.savefig('./data/find_gamma/weight_alpha='+al+'.pdf',bbox_inches='tight',pad_inches=0.0,dpi = 500)

def compare_method():
    gs_val = os.listdir(config.get('gs_samples_val'))
    mab_val = os.listdir(config.get('mab_samples_val'))
    kreuk_val = os.listdir(config.get('kreuk_samples_val'))
    malware_val = os.listdir(config.get('val'))
    benign_val = os.listdir(config.get('benign_samples'))
    beta = -0.045
    gamma = 40
    with open('./data/compare.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type','our','gs','mab','kreuk'])
    i =0
    our_detect = 0
    gs_detect = 0
    mab_detect = 0
    kreuk_detect = 0
    for file in gs_val:
        if i == 1000:
            i=0
            break
        file_path = "/home/cjh/adv_samples_retrain/gs/val/malware/"+file
        pred , score , fv= guarded_detect_file(file_path,[6,12,18,36,83,95,102,113,124],beta,gamma,weight)
        our_detect = our_detect + pred
        pred , score ,fv = detect_file(clf1,file_path)
        gs_detect = gs_detect + pred
        pred , score ,fv = detect_file(clf2,file_path)
        mab_detect = mab_detect + pred
        pred , score , fv = detect_file(clf3,file_path)
        kreuk_detect = kreuk_detect + pred
        i = i+1
    with open('./data/compare.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gs',str(our_detect/1000),str(gs_detect/1000),str(mab_detect/1000),str(kreuk_detect/1000)])
    print("gs_done!")
    our_detect = 0
    gs_detect = 0
    mab_detect = 0
    kreuk_detect = 0
    for file in mab_val:
        if i == 1000:
            i=0
            break
        file_path = "/home/cjh/adv_samples_retrain/mab/val/malware/"+file
        pred , score , fv= guarded_detect_file(file_path,[6,12,18,36,83,95,102,113,124],beta,gamma,weight)
        our_detect = our_detect + pred
        pred , score ,fv = detect_file(clf1,file_path)
        gs_detect = gs_detect + pred
        pred , score ,fv = detect_file(clf2,file_path)
        mab_detect = mab_detect + pred
        pred , score , fv = detect_file(clf3,file_path)
        kreuk_detect = kreuk_detect + pred
        i = i+1
    with open('./data/compare.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mab',str(our_detect/1000),str(gs_detect/1000),str(mab_detect/1000),str(kreuk_detect/1000)])
    print("mab done!")
    our_detect = 0
    gs_detect = 0
    mab_detect = 0
    kreuk_detect = 0
    for file in kreuk_val:
        if i == 1000:
            i=0
            break
        file_path = "/home/cjh/adv_samples_retrain/kreuk/val/malware/"+file
        pred , score , fv= guarded_detect_file(file_path,[6,12,18,36,83,95,102,113,124],beta,gamma,weight)
        our_detect = our_detect + pred
        pred , score ,fv = detect_file(clf1,file_path)
        gs_detect = gs_detect + pred
        pred , score ,fv = detect_file(clf2,file_path)
        mab_detect = mab_detect + pred
        pred , score , fv = detect_file(clf3,file_path)
        kreuk_detect = kreuk_detect + pred
        i = i+1
    with open('./data/compare.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kreuk',str(our_detect/1000),str(gs_detect/1000),str(mab_detect/1000),str(kreuk_detect/1000)])
    print("kreuk done!")
    our_detect = 0
    gs_detect = 0
    mab_detect = 0
    kreuk_detect = 0
    for file in malware_val:
        if i == 1000:
            i=0
            break
        file_path = "../Val/"+file
        pred , score , fv= guarded_detect_file(file_path,[6,12,18,36,83,95,102,113,124],beta,gamma,weight)
        our_detect = our_detect + pred
        pred , score ,fv = detect_file(clf1,file_path)
        gs_detect = gs_detect + pred
        pred , score ,fv = detect_file(clf2,file_path)
        mab_detect = mab_detect + pred
        pred , score , fv = detect_file(clf3,file_path)
        kreuk_detect = kreuk_detect + pred
        i = i+1
    with open('./data/compare.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['malware',str(our_detect/1000),str(gs_detect/1000),str(mab_detect/1000),str(kreuk_detect/1000)])
    print("malware done!")
    our_detect = 0
    gs_detect = 0
    mab_detect = 0
    kreuk_detect = 0
    for file in benign_val:
        if i == 1000:
            i=0
            break

        file_path = "../Benign/benign/"+file
        pred , score , fv= guarded_detect_file(file_path,[6,12,18,36,83,95,102,113,124],beta,gamma,weight)
        our_detect = our_detect + pred
        pred , score ,fv = detect_file(clf1,file_path)
        gs_detect = gs_detect + pred
        pred , score ,fv = detect_file(clf2,file_path)
        mab_detect = mab_detect + pred
        pred , score , fv = detect_file(clf3,file_path)
        kreuk_detect = kreuk_detect + pred
        i = i+1
    with open('./data/compare.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['benign',str(1-our_detect/1000),str(1-gs_detect/1000),str(1-mab_detect/1000),str(1-kreuk_detect/1000)])
    print("benign done!")
detectPath(clf,config.get('benign_samples'))
# findParaBeta(weight,afLists)
# findParaGamma(weight,afLists)
# detectPath("../mab_samples/output/evasive/")
# compare_method()
# getFCweight(clf)