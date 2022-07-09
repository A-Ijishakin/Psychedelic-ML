import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from fit_model import GradAlign 

Data = GradAlign.aligned_[0][:, 0]        
for idx in range(1, 20):
    Data += GradAlign.aligned_[idx][:, 0]
    
MeanData = Data/20 #average value of DMT gradients

DataList = []

for i in range(20):
    for val in GradAlign.aligned_[i][:, 0]:
        DataList.append(val)

#create Histograms 
bins = np.linspace(-0.125, 0.125, 100)

plt.hist(DataList, bins, alpha = 0.5, label = 'DMT', color = 'blue')
plt.xlabel('Gradient Scores')
plt.ylabel('Frequency')
plt.title('Gradient Distribution') 
plt.legend(loc = 'upper right') 
plt.show()  

differences = [] 

for i in range(20): 
        differences.append(np.max(GradAlign.aligned_[i][:, 0]) - np.min(GradAlign.aligned_[i][:, 0]))   

figure = plt.figure(figsize = (10, 17))  

plt.xlabel('Gradients')
plt.ylabel('Gradient Span') 
plt.title('Principal Gradient') 

c = 'red' 

data = pd.DataFrame(differences) 

plt.boxplot(data, notch=None, sym= '', vert=True, whis=None, positions=None, widths=None, patch_artist=True, bootstrap=1000)