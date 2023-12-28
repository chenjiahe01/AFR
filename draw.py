import matplotlib.pyplot as plt
import pandas as pd

for type in ['beta','gamma']:
    if type == 'beta':
        x = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,-0.01,-0.03,-0.05,-0.07,-0.09,-0.1]
        s = 'β'
        x_lim = [1,-0.1]
    else:
        x = [1,10,20,30,40,50,60,70,80,90,100]
        s = 'γ'
        x_lim = [1,100]
    for alpha in ["0.9","0.8","0.7","0.6","0.5"]:
        data = pd.read_csv('data/find_'+type+'/alpha='+alpha+'_find_'+type+'.csv')
        drs_kreuk = data['dr_kreuk_samples'].tolist()
        drs_gs = data['dr_gs_samples'].tolist()
        drs_mab = data['dr_mab_samples'].tolist()
        drs_benign = data['dr_benign_samples'].tolist()
        drs_mal = data['dr_mal_samples'].tolist()
        # print(drs_kreuk)
        fig, ax = plt.subplots(1, 1,figsize=(20, 20))
        ax.set_xlim(x_lim)
        print(x)
        ax.plot(x, drs_kreuk, linewidth= 2, color="red", label='Kreuk'  )
        ax.plot(x, drs_gs, linewidth= 2,color="black", label='Gamma Sections')
        ax.plot(x, drs_mab, linewidth= 2,color="blue", label='MAB')
        ax.plot(x, drs_mal, linewidth= 2,color="green", label='Malware')
        ax.plot(x, drs_benign, linewidth= 2,color="orange", label='Benign')
        ax.set_xlabel(s,fontsize=50)  # 设置横轴标签  
        ax.set_ylabel('detection rate',fontsize=50)  # 设置纵轴标签  
        ax.tick_params(axis='both', which='major', labelsize=50)
        legend=ax.legend(fontsize=36,bbox_to_anchor=(0.5, 0.1, 0.5, 0.6))
        plt.savefig('./data/find_'+type+'/weight_alpha='+alpha+'.pdf',bbox_inches='tight',pad_inches=0.0,dpi = 500)