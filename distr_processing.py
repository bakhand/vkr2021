from os import listdir
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%%

files = [f for f in listdir(pathlib.Path().absolute()) if "t.pickle" in f ]


raw_times = {}


for f in files:
    with open(f, 'rb') as f1:
       data_new = pickle.load(f1)
    raw_times[f[0:-8]] = data_new
    
    
raw_lensSWD = {}

for key in raw_times.keys():
    lensSWD = raw_times[key]['stops'] - raw_times[key]['starts']
    raw_lensSWD[key] = lensSWD

#%%собираю длины в кучу для каждого животного
lensSWD = {}
lensSWD["KMA"] = raw_lensSWD["KMA"]
lensSWD["KM2"] = raw_lensSWD["KM2"]
lensSWD["KM2_NEW"] = raw_lensSWD["KM2_NEW"]
lensSWD["NULL1_NEW"] = raw_lensSWD["NULL1_NEW"]
lensSWD["KM1"] = np.concatenate((raw_lensSWD["KM1_0"],raw_lensSWD["KM1_1"],raw_lensSWD["KM1_2"]))
lensSWD["NULL1"] = np.concatenate((raw_lensSWD["NULL1_0"], raw_lensSWD["NULL1_1"], raw_lensSWD["NULL1_3"]))





#%%





#%%считаю chtly

print(lensSWD["KMA"].mean()/1000)
print(lensSWD["KM2"].mean()/1000)
print(lensSWD["KM1"].mean()/1000)
print(lensSWD["NULL1"].mean()/1000)

print(np.median(lensSWD["KMA"])/1000)
print(np.median(lensSWD["KM2"])/1000)
print(np.median(lensSWD["KM1"])/1000)
print(np.median(lensSWD["NULL1"])/1000)


print(np.std(lensSWD["KMA"])/1000)
print(np.std(lensSWD["KM2"])/1000)
print(np.std(lensSWD["KM1"])/1000)
print(np.std(lensSWD["NULL1"])/1000)



print(np.percentile(lensSWD["KMA"], (25,75))/1000)
print(np.percentile(lensSWD["KM2"], (25,75))/1000)
print(np.percentile(lensSWD["KM1"], (25,75))/1000)
print(np.percentile(lensSWD["NULL1"], (25,75))/1000)



#%%распределения на отдельных графиках



ncols = 4
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5), sharey='row')
fig.suptitle('Плотности распределения длин РПВ (до 30 сек)', size = 15)
fig.text(0.5, 0.04, 'Продолжительность, секунды', ha='center', va='center', size = 14)


fig.text(0.225, 0.75, 'КМ А', ha='center', va='center')
fig.text(0.425, 0.75, 'КМ 1', ha='center', va='center')
fig.text(0.625, 0.75, 'КМ 2', ha='center', va='center')
fig.text(0.850, 0.75, 'НОЛЬ 1', ha='center', va='center')


ax1 = axes[0]

ax1.hist(lensSWD["KMA"]/1000,bins = 50, range = (1,30), density = True)
ax1.axvline(x=lensSWD["KMA"].mean()/1000, c ='r', ymin=0, ymax=1)
ax1.axvline(x=np.median(lensSWD["KMA"])/1000, c ='r', ls = '--', ymin=0, ymax=1)

shape,loc,scale = stats.lognorm.fit(lensSWD["KMA"]/1000)
x = np.linspace(1, 30, 200)
pdf = stats.lognorm.pdf(x, shape, loc, scale)
ax1.plot(x, pdf, 'k')
print(shape,loc,scale)



ax1 = axes[1]
ax1.hist(lensSWD["KM1"]/1000,bins = 50, range = (1,30),density = True)
ax1.axvline(x=lensSWD["KM1"].mean()/1000, c ='r', ymin=0, ymax=1)
ax1.axvline(x=np.median(lensSWD["KM1"])/1000, c ='r', ls = '--', ymin=0, ymax=1)

shape,loc,scale = stats.lognorm.fit(lensSWD["KM1"]/1000)
x = np.linspace(1, 30, 200)
pdf = stats.lognorm.pdf(x, shape, loc, scale)
ax1.plot(x, pdf, 'k')
print(shape,loc,scale)



ax1 = axes[2]
ax1.hist(lensSWD["KM2"]/1000,bins = 50, range = (1,30),density = True)
ax1.axvline(x=lensSWD["KM2"].mean()/1000, c ='r', ymin=0, ymax=1)
ax1.axvline(x=np.median(lensSWD["KM2"])/1000, c ='r', ls = '--', ymin=0, ymax=1)
shape,loc,scale = stats.lognorm.fit(lensSWD["KM2"]/1000)
x = np.linspace(1, 30, 200)
pdf = stats.lognorm.pdf(x, shape, loc, scale)
ax1.plot(x, pdf, 'k')
print(shape,loc,scale)


ax1 = axes[3]
ax1.hist(lensSWD["NULL1"]/1000,bins = 50, range = (1,30),density = True)
ax1.axvline(x=lensSWD["NULL1"].mean()/1000, c ='r', ymin=0, ymax=1)
ax1.axvline(x=np.median(lensSWD["NULL1"])/1000, c ='r', ls = '--', ymin=0, ymax=1)
shape,loc,scale = stats.lognorm.fit(lensSWD["NULL1"]/1000)
print(shape,loc,scale)

x = np.linspace(1, 30, 200)
pdf = stats.lognorm.pdf(x, shape, loc, scale)
ax1.plot(x, pdf, 'k')

fig.show()

#%%Колмогоров смирнов








#%%

fig = plt.figure()
ax = fig.add_subplot(111)


ax.hist(lensSWD["KMA"]/1000,bins = 50, density = True)
ax.hist(lensSWD["KM1"]/1000,bins = 50, density = True)
ax.hist(lensSWD["KM2"]/1000,bins = 50, density = True)
ax.hist(lensSWD["NULL1"]/1000,bins = 50, density = True)


plt.show()




#%%собираю сырые длины

raw_lensFree = {}



for key in raw_times.keys():
    lensFree = raw_times[key]['starts'][1:] - raw_times[key]['stops'][0:-1]
    raw_lensFree[key] = lensFree





#%%собираю длины в кучу для каждого животного
lensFree= {}
lensFree["KMA"] = raw_lensFree["KMA"]
lensFree["KM2"] = raw_lensFree["KM2"]
lensFree["KM1"] = np.concatenate((raw_lensFree["KM1_0"],raw_lensFree["KM1_1"],raw_lensFree["KM1_2"]))
lensFree["NULL1"] = np.concatenate((raw_lensFree["NULL1_0"], raw_lensFree["NULL1_1"], raw_lensFree["NULL1_3"]))
 #%% строю гистограммки длин


ncols = 4
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5), sharey='row')
fig.suptitle('Плотности распределения промежутков времени без РПВ до 10 минут', size = 15)
fig.text(0.5, 0.04, 'Продолжительность, минуты', ha='center', va='center', size = 14)


fig.text(0.225, 0.75, 'КМ А', ha='center', va='center')
fig.text(0.425, 0.75, 'КМ 1', ha='center', va='center')
fig.text(0.625, 0.75, 'КМ 2', ha='center', va='center')
fig.text(0.850, 0.75, 'НОЛЬ 1', ha='center', va='center')


ax1 = axes[0]

ax1.hist(lensFree["KMA"]/60000,bins = 50, range = (0,10), density = True)

ax1 = axes[1]
ax1.hist(lensFree["KM1"]/60000,bins = 50, range = (0,10),density = True)


ax1 = axes[2]
ax1.hist(lensFree["KM2"]/60000,bins = 50, range = (0,10),density = True)



ax1 = axes[3]
ax1.hist(lensFree["NULL1"]/60000,bins = 50, range = (0,10),density = True)


fig.show()


#%%


from scipy.stats import beta


KMA_len = 260673664


beta_starts = lensFree["NULL1_NEW"]/KMA_len

mean, var, skew, kurt = stats.beta.stats(1, len(beta_starts), moments='mvsk')


fig, ax = plt.subplots(1, 1)

a, b = 1, len(beta_starts)
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(beta.ppf(0.01, a, b),
                beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),
       'r-', lw=5, alpha=0.6, label='beta pdf')

rv = beta(a, b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')



vals = beta.ppf([0.001, 0.5, 0.999], a, b)
np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, a, b))

ax.hist(beta_starts, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()



ncols = 4
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
fig.suptitle('Плотности распределения промежутков времени без РПВ до 10 минут', size = 15)
fig.text(0.5, 0.04, 'Продолжительность, минуты', ha='center', va='center', size = 14)


fig.text(0.225, 0.75, 'КМ А', ha='center', va='center')
fig.text(0.425, 0.75, 'КМ 1', ha='center', va='center')
fig.text(0.625, 0.75, 'КМ 2', ha='center', va='center')
fig.text(0.850, 0.75, 'НОЛЬ 1', ha='center', va='center')


ax1 = axes[0]

ax1.hist(6000000/lensFree["KMA"],bins = 5,  density = True)

ax1 = axes[1]
ax1.hist(6000000/lensFree["KM1"],bins = 5, density = True)


ax1 = axes[2]
ax1.hist(6000000/lensFree["KM2"],bins = 5, density = True)



ax1 = axes[3]
ax1.hist( 6000000/lensFree["NULL1"], bins = 5 ,density = True)


fig.show()


#%%
files = [f for f in listdir(pathlib.Path().absolute()) if "d.pickle" in f ]


raw_shifts = {}


for f in files:
    with open(f, 'rb') as f1:
       data_new = pickle.load(f1)
    raw_shifts[f[0:-8]] = data_new
    

#%%считаю параметры для КМ2 и строю для каждого часа

KM2_len = 260673664


seps = [0]
step = 3600*1000


while seps[-1]<(KM2_len - step):
    seps.append(seps[-1]+step)



last_len = (KM2_len - seps[-1])
seps.append(KM2_len)


pSWI = np.zeros(len(seps)-1)
nSWD = np.zeros(len(seps)-1)




KM2_starts = raw_times['NULL1_NEW']['starts']
KM2_lens = lensSWD["NULL1_NEW"] 


counter = 0

for i in range(len(KM2_starts)):
     counter = KM2_starts[i]//step
     pSWI[counter]+=KM2_lens[i]
     nSWD[counter]+=1



pSWI = np.array(pSWI)*100.0/step

pSWI[-1] = (pSWI[-1]*step)/last_len

hour_ind = list(range(1,len(seps)))

from matplotlib.patches import Rectangle


lights_start = -2.36


ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
fig.suptitle('ПВИ для каждого часа записи. Крыса НОЛЬ 1 НОВ', size = 15)

# ax.add_patch(Rectangle((lights_start, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+24, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+48, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+72, 0), 12, 3, facecolor = 'black', alpha = 0.2))
ax.bar(hour_ind, pSWI)
ax.set_xlim((1,70))
ax.set_ylim((0,3))
ax.set_xlabel('Час от начала записи', size = 14)
ax.set_ylabel('ПВИ, %', size = 14)
fig.show()

#%%


KM2_pSWI = pSWI
KM2_nSWD = nSWD

ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
fig.suptitle('Зависмость количества РПВ от ПВИ. Крыса КМ 2', size = 15)


ax.plot(KM2_pSWI, KM2_nSWD, 'ro')
ax.set_xlabel('ПВИ, %', size = 14)
ax.set_ylabel('РПВ, событий в час', size = 14)
fig.show()






#%%теперь строю то же самое для КМ 2 с вариациями по времени

KM2_len = 260673664

seps = [0]
step = 3600*1000*1


while seps[-1]<(KM2_len - step):
    seps.append(seps[-1]+step)



last_len = (KM2_len - seps[-1])
seps.append(KM2_len)


pSWI = np.zeros(len(seps)-1)
nSWD = np.zeros(len(seps)-1)




KM2_starts = raw_times['KM2_NEW']['starts']
KM2_lens = lensSWD["KM2_NEW"] 


counter = 0

for i in range(len(KM2_starts)):
     counter = KM2_starts[i]//step
     pSWI[counter]+=KM2_lens[i]
     nSWD[counter]+=1



pSWI = np.array(pSWI)*100.0/step

pSWI[-1] = (pSWI[-1]*step)/last_len

hour_ind = list(range(1,len(seps)))

from matplotlib.patches import Rectangle


lights_start = -2.36


ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
fig.suptitle('ПВИ последовательных интервалов записи. \n Крыса НОЛЬ 1 НОВ, интервал 1ч', size = 15)

# ax.add_patch(Rectangle((lights_start, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+24, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+48, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+72, 0), 12, 3, facecolor = 'black', alpha = 0.2))
ax.bar(hour_ind, pSWI)

ax.axhline(y=np.mean(pSWI), xmin=0, xmax=1, c= 'r')
ax.axhline(y=np.median(pSWI), xmin=0, xmax=1, c= 'r', ls = '--')

ax.set_xlim((1,max(hour_ind)))
ax.set_ylim((0,3))
ax.set_xlabel('Интервал от начала записи', size = 14)
ax.set_ylabel('ПВИ, %', size = 14)
fig.show()






#%%Теперь для КМ 1 соберу штучки


raw_lensSWD = {}
raw_starts = {}


for key in raw_times.keys():
    lensSWD = raw_times[key]['stops'] - raw_times[key]['starts']
    raw_starts[key] = raw_times[key]['starts']
    raw_lensSWD[key] = lensSWD


Kd = raw_shifts['KM1']
KM1_len = (Kd['third_block_end'] - Kd['first_block_start']).astype('int')
shift1 = (Kd['second_block_start'] - Kd['first_block_start']).astype('int')*1000
shift2 = (Kd['third_block_start'] - Kd['first_block_start']).astype('int')*1000

KM1_lens = np.concatenate((raw_lensSWD["KM1_0"],raw_lensSWD["KM1_1"],raw_lensSWD["KM1_2"]))
KM1_starts = np.concatenate((raw_starts["KM1_0"], raw_starts["KM1_1"] + shift1,raw_starts["KM1_2"] + shift2))




#%%строю основной график для КМ1


seps = [0]
step = 3600*1000


while seps[-1]<(KM1_len - step):
    seps.append(seps[-1]+step)



last_len = (KM1_len - seps[-1])
seps.append(KM1_len)


pSWI = np.zeros(len(seps)-1)
nSWD = np.zeros(len(seps)-1)


counter = 0

for i in range(len(KM1_starts)):
     counter = KM1_starts[i]//step
     pSWI[counter]+=KM1_lens[i]
     nSWD[counter]+=1



pSWI = np.array(pSWI)*100.0/step

pSWI[-1] = np.mean(pSWI)


hour_ind = list(range(1,len(seps)))

from matplotlib.patches import Rectangle


lights_start = Kd['light_off']/(3600*1000)


loss1_start = (Kd['second_block_end']-Kd['first_block_start']).astype('int')/(3600*1000)
loss1_len = Kd['second_delta'].astype('int')/(3600*1000)

ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
fig.suptitle('ПВИ для каждого часа записи. Крыса КМ 1', size = 15)

ax.add_patch(Rectangle((lights_start, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+24, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+48, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+72, 0), 12, 17, facecolor = 'black', alpha = 0.2))


ax.add_patch(Rectangle((loss1_start, 0), loss1_len, 17, facecolor = 'w', hatch = '//'))

ax.bar(hour_ind, pSWI)
ax.set_xlim((1,max(hour_ind)+1))

ax.set_ylim((0,17))
ax.set_xlabel('Час от начала записи', size = 14)
ax.set_ylabel('ПВИ, %', size = 14)
fig.show()


#%%
KM1_pSWI = pSWI
KM1_nSWD = nSWD

ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
fig.suptitle('Зависмость количества РПВ от ПВИ. Крыса КМ 1', size = 15)


ax.plot(KM1_pSWI, KM1_nSWD, 'ro')
ax.plot(KM2_pSWI, KM2_nSWD, 'bo')
ax.set_xlabel('ПВИ, %', size = 14)
ax.set_ylabel('РПВ, событий в час', size = 14)
fig.show()

#%%Крыса НОЛЬ1


raw_lensSWD = {}
raw_starts = {}


for key in raw_times.keys():
    lensSWD = raw_times[key]['stops'] - raw_times[key]['starts']
    raw_starts[key] = raw_times[key]['starts']
    raw_lensSWD[key] = lensSWD


Kd = raw_shifts['NULL1']
NULL1_len = (Kd['third_block_end'] - Kd['first_block_start']).astype('int')
shift1 = (Kd['second_block_start'] - Kd['first_block_start']).astype('int')*1000
shift2 = (Kd['third_block_start'] - Kd['first_block_start']).astype('int')*1000

NULL1_lens = np.concatenate((raw_lensSWD["NULL1_0"],raw_lensSWD["NULL1_1"],raw_lensSWD["NULL1_3"]))
NULL1_starts = np.concatenate((raw_starts["NULL1_0"], raw_starts["NULL1_1"] + shift1,raw_starts["NULL1_3"] + shift2))



seps = [0]
step = 3600*1000


while seps[-1]<(NULL1_len - step):
    seps.append(seps[-1]+step)



last_len = (NULL1_len - seps[-1])
seps.append(NULL1_len)


pSWI = np.zeros(len(seps)-1)
nSWD = np.zeros(len(seps)-1)


counter = 0

for i in range(len(NULL1_starts)):
     counter = NULL1_starts[i]//step
     pSWI[counter]+=NULL1_lens[i]
     nSWD[counter]+=1



pSWI = np.array(pSWI)*100.0/step

pSWI[-1] = np.mean(pSWI)


hour_ind = list(range(1,len(seps)))

from matplotlib.patches import Rectangle


lights_start = Kd['light_on']/(3600*1000) - 12


loss1_start = (Kd['first_block_end']-Kd['first_block_start']).astype('int')/(3600*1000)
loss1_len = Kd['first_delta'].astype('int')/(3600*1000)

loss2_start = (Kd['second_block_end']-Kd['first_block_start']).astype('int')/(3600*1000)+2
loss2_len = Kd['second_delta'].astype('int')/(3600*1000)-2


ncols = 1
nrows = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
fig.suptitle('ПВИ для каждого часа записи. Крыса НОЛЬ 1', size = 15)


ax.add_patch(Rectangle((loss1_start, 0), loss1_len, 17, facecolor = 'w', hatch = '//'))
ax.add_patch(Rectangle((loss2_start, 0), loss2_len, 17, facecolor = 'w', hatch = '//'))


ax.add_patch(Rectangle((lights_start, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+24, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+48, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+72, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+72+24, 0), 12, 17, facecolor = 'black', alpha = 0.2))
ax.add_patch(Rectangle((lights_start+72+24+24, 0), 12, 17, facecolor = 'black', alpha = 0.2))


ax.bar(hour_ind, pSWI)
ax.set_xlim((1,max(hour_ind)+1))

ax.set_ylim((0,1))
ax.set_xlabel('Час от начала записи', size = 14)
ax.set_ylabel('ПВИ, %', size = 14)
fig.show()


#%%

NULL1_pSWI = pSWI
NULL1_nSWD = nSWD

ncols = 1
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
fig.suptitle('Зависимость количества РПВ от ПВИ', size = 15)


ax[0].plot(KM1_pSWI, KM1_nSWD, 'ro')
ax[0].plot(KM2_pSWI, KM2_nSWD, 'bo')
ax[0].plot(NULL1_pSWI, NULL1_nSWD, 'go')
ax[0].legend()

ax[0].set_ylabel('РПВ, событий в час', size = 14)


ax[1].plot(KM1_pSWI, KM1_nSWD, 'ro')
ax[1].plot(KM2_pSWI, KM2_nSWD, 'bo')
ax[1].plot(NULL1_pSWI, NULL1_nSWD, 'go')

ax[1].set_xlabel('ПВИ, %', size = 14)
ax[1].set_ylabel('РПВ, событий в час', size = 14)
ax[1].set_xlim((0, 2.5))
ax[1].set_ylim((0, 20))

ax[0].plot(KM1_pSWI, KM1_nSWD, 'ro')
ax[0].plot(KM2_pSWI, KM2_nSWD, 'bo')
ax[0].plot(NULL1_pSWI, NULL1_nSWD, 'go')

fig.show()



# #%%

# KM2_len = 250426111


# seps = [0]
# step = 3600*1000


# while seps[-1]<(KM2_len - step):
#     seps.append(seps[-1]+step)



# last_len = (KM2_len - seps[-1])
# seps.append(KM2_len)


# pSWI = np.zeros(len(seps)-1)
# nSWD = np.zeros(len(seps)-1)




# KM2_starts = raw_times['KM2']['starts']
# KM2_lens = lensSWD["KM2"] 


# counter = 0

# for i in range(len(KM2_starts)):
#      counter = KM2_starts[i]//step
#      pSWI[counter]+=KM2_lens[i]
#      nSWD[counter]+=1



# pSWI = np.array(pSWI)*100.0/step

# pSWI[-1] = (pSWI[-1]*step)/last_len

# hour_ind = list(range(1,len(seps)))

# from matplotlib.patches import Rectangle


# lights_start = -2.36


# ncols = 1
# nrows = 1
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
# fig.suptitle('ПВИ для каждого часа записи. \n Крыса КМ 2', size = 15)

# ax.add_patch(Rectangle((lights_start, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+24, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+48, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.add_patch(Rectangle((lights_start+72, 0), 12, 3, facecolor = 'black', alpha = 0.2))
# ax.bar(hour_ind, pSWI)
# ax.set_xlim((1,70))
# ax.set_ylim((0,3))
# ax.set_xlabel('Час от начала записи', size = 14)
# ax.set_ylabel('ПВИ, %', size = 14)
# fig.show()




