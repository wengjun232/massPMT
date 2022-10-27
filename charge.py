#charge ana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob,os
import argparse
import h5py
import csv
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
filename = glob.glob('data/'+PMT+'/*SPE*')
#print(filename)
#path = '/home/wengjun/pmt/PM2208-4005-1/data-SPE-rawdata-2022-8-25-8-1-0.xls'
#PMT = 'PM2208-4005'
path = filename[0]
#print(path)
data = pd.read_csv(path, sep='\t')
rawdata = np.array(data[PMT])
x = np.arange(len(rawdata))
ground = np.argmax(rawdata)
cut = 0.25*1.6/0.025
signal = np.copy(rawdata)
signal[0:(int(cut)+ground)] = 0
height = np.max(signal)
uc1 = np.argmax(signal)
#rawdata = np.array(data[PMT])
val0 = np.argmin(rawdata[ground:uc1])
val = val0+ground
a=0.06
b=0.25
popt1, pcov1 = curve_fit(gau,0.025*x[(uc1-int(0.35*(uc1-ground))):(uc1+int(0.35*(uc1-ground)))] ,
                         rawdata[(uc1-int(0.35*(uc1-ground))):(uc1+int(0.35*(uc1-ground)))],p0=[0.025*uc1,1,100])
popt2, pcov2 = curve_fit(paraboli,0.025*x[(val-int(a*(uc1-ground))):(val+int(b*(uc1-ground)))] ,
                         rawdata[(val-int(a*(uc1-ground))):(val+int(b*(uc1-ground)))],
                         p0=[300,0.025*val,np.min(rawdata[ground:uc1])],
                         bounds=([0.1,(0.025*(val-5)),np.min(rawdata[ground:uc1])],[10000,(0.025*(val+5)),np.min(rawdata[ground:uc1])+10]))
gain = ( popt1[0] -0.025*ground)*10**(-12)/(1.6*10**(-19))
PV = np.max(gau(0.025*x[(uc1-int(0.35*(uc1-ground))):(uc1+int(0.35*(uc1-ground)))],*popt1))/popt2[2]
if not os.path.exists('result/'+PMT):
    os.makedirs('result/'+PMT)
plt.plot(0.025*x, rawdata,color='dimgrey')
plt.plot(0.025*x[(uc1-int(0.35*(uc1-ground))):(uc1+int(0.35*(uc1-ground)))], 
         gau(0.025*x[(uc1-int(0.35*(uc1-ground))):(uc1+int(0.35*(uc1-ground)))], *popt1),
         'r',label='gau: $\mu$=%5.3f \n gain:%5.3f \n p\V:%5.3f' % (popt1[0],gain,PV)
plt.plot(0.025*x[(val-int(a*(uc1-ground))):(val+int(b*(uc1-ground)))], 
         paraboli(0.025*x[(val-int(a*(uc1-ground))):(val+int(b*(uc1-ground)))], *popt2), 
         'b',label='parabolic : valley=%5.3f' % popt2[2])
plt.axvline(x=0.025*ground,ls='--',color='black')
plt.legend()
plt.xlim(0,0.025*500)
plt.ylim(0,height+100)
plt.title("charge spectrum of "+PMT)
plt.xlabel("charge/pC")
plt.ylabel("count")
#plt.show()
plt.savefig('result/'+PMT+'/charge.pdf')
with open('pmtid_charge.csv','a',newline='') as f1:
    writer1 = csv.writer(f1)
    writer1.writerow([PMT])
with open('gain.csv','a',newline='') as f2:
    writer2 = csv.writer(f2)
    writer2.writerow([gain])
with open('PV.csv','a',newline='') as f3:
    writer3 = csv.writer(f3)
    writer3.writerow([PV])
#print(gain)
#print(PMT)
#plt.savefig("spechargeFit.svg")
