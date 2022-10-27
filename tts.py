#tts ana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob,os,csv
import argparse
import h5py
from scipy.optimize import curve_fit
def gau(x,mu,sig,A):
    return A*(np.exp(-(x - mu) ** 2 / (2 * sig ** 2)))
def paraboli(x,a,b,c):
    return a*(x-b)**2+c
psr = argparse.ArgumentParser()
psr.add_argument("-i", dest = 'ipt',help="input")
args = psr.parse_args()
num = int(args.ipt)
names = os.listdir('data')
PMT = names[num]
print(PMT)
filename = glob.glob('data/'+PMT+'/*TTS*')
if len(filename)>0:
    print(filename)
#path = '/home/wengjun/pmt/PM2208-4005-1/data-SPE-rawdata-2022-8-25-8-1-0.xls'
#PMT = 'PM2208-4005'
    path = filename[0]
    print(path)
    data = pd.read_csv(path, sep='\t')
    #rawdata[rawdata=='PA-']=0
    #rawdata = np.array(rawdata,dtype=int)
    rawdata = np.array(data[PMT])
    rawdata[rawdata=='PA-']=0
    rawdata = np.array(rawdata,dtype=int)
    x = 0.1*np.arange(len(rawdata))
    peak = np.argmax(rawdata)
    if np.max(rawdata)>10:
        popt1, pcov1 = curve_fit(gau,x[(peak-30):(peak+30)],rawdata[(peak-30):(peak+30)],p0=[peak*0.1,1,800])
        plt.plot(x[peak-30:peak+30],rawdata[peak-30:peak+30],label="data")
        plt.plot(x[peak-30:peak+30],gau(x[peak-30:peak+30],*popt1),'r',label="fit:$\sigma$=%5.3f"% popt1[1])
        plt.xlabel('TT/ns')
        plt.ylabel('count')
        plt.title('TT of '+PMT)
        plt.legend()
        plt.savefig('result/'+PMT+'/TT.pdf')
        TTS = 2*np.sqrt(2*np.log(2))*popt1[1]
        with open('TTS.csv','a',newline='') as f3:
            writer3 = csv.writer(f3)
            writer3.writerow([TTS])
        with open('pmt_tts.csv','a',newline='') as f4:
            writer4 = csv.writer(f4)
            writer4.writerow([PMT])
#print(TTS)
