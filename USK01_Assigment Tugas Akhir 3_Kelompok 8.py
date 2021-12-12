#!/usr/bin/env python
# coding: utf-8

# ## EKSPLORASI DAN PERSIAPAN DATA

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#import semua data
ipm=pd.read_csv('ipm.csv')
uhh=pd.read_csv('uhh.csv')
ppp=pd.read_csv('ppp.csv')
hls=pd.read_csv('hls.csv')
rls=pd.read_csv('rls.csv')

print(ipm.head(2))
print ()
print(uhh.head(2))
print ()
print(ppp.head(2))
print ()
print(hls.head(2))
print ()
print(rls.head(2))


# In[3]:


#ubah tabel ipm
df1=ipm.melt(id_vars="Provinsi",
             var_name="Tahun",
             value_name="IPM")
df1


# In[4]:


#ubah tabel UHH
df2=uhh.melt(id_vars="Provinsi",
             var_name="Tahun",
             value_name="UHH")
df2


# In[5]:


#ubah tabel PPP
df3=ppp.melt(id_vars="Provinsi",
             var_name="Tahun",
             value_name="PPP")

df3


# In[6]:


#ubah tabel HLS
df4=hls.melt(id_vars="Provinsi",
             var_name="Tahun",
             value_name="HLS")

df4


# In[7]:


#ubah tabel RLS
df5=rls.melt(id_vars="Provinsi",
             var_name="Tahun",
             value_name="RLS")

df5


# In[8]:


from functools import reduce

##Name of a column in all dataframes is 'Provinsi and Tahun'
data_frames=[df1,df2,df3,df4,df5]
data = reduce(lambda  left,right: pd.merge(left,right,on=['Provinsi','Tahun'],
                                            how='outer'), data_frames)
data.head(10)


# Keterangan variabel:
# - IPM = Indeks Pembangunan Manusia (Variabel Dependen/Y)
# - UHH = Usia Harapan Hidup (Variabel Independen/X1)
# - PPP = Pendapatan per Kapita (Variabel Independen/X2)
# - HLS = Harapan Lama Sekolah (Variabel Independen/X3)
# - RLS = Rata-rata Lama Sekolah (Variabel Independen/X4)

# In[9]:


data=data.drop(['Provinsi','Tahun'], axis = 1)


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


data=data.dropna()


# ### CEK DATA SETELAH PROSES CLEANSING

# In[14]:


data.info()


# In[15]:


data.describe()


# In[16]:


data.isnull().sum()


# ### Eksplorasi Data

# In[17]:


# import library matplotlib
import matplotlib.pyplot as plt

# import library seaborn
import seaborn as sns

# me non aktifkan peringatan pada python dengan import warning -> 'ignore'
import warnings
warnings.filterwarnings("ignore")


# In[18]:


corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.title('Correlation Matrix', loc='center', fontsize=15)
plt.show()


# Berdasarkan hasil yang telah diperoleh dapat dilihat nilai korelasi antar variabel. Perolehan korelasi paling kuat yaitu pada variabel IPM dengan variabel PPP senilai 0.87, diikuti oleh variabel IPM dengan variabel RLS senilai 0.83, variabel IPM dengan variabel UHH senilai 0.77 dan ariabel IPM dengan variabel HLS senilai 0.65 . 

# In[19]:


#Plot 
fig, ax=plt.subplots(2,2,figsize=(20,6))
sns.scatterplot(data['UHH'],data['IPM'], ax=ax[0,0])
sns.scatterplot(data['PPP'],data['IPM'], ax=ax[0,1])
sns.scatterplot(data['HLS'],data['IPM'], ax=ax[1,0])
sns.scatterplot(data['RLS'],data['IPM'], ax=ax[1,1])


# Berdasarkan scatterplot di atas dapat diketahui bahwa variabel IPM dengan variabel UHH, PPP, HLS, dan RLS memiliki hubungan atau korelasi yang positif di mana jika nilai UHH atau PPP atau HLS atau RLS meningkat maka nilai IPM juga akan meningkat, begitu pula sebaliknya

# ##### Klasifikasi IPM menurut BPS(Badan Pusat Statistik) dibagi menjadi 4 kategori yaitu :
# - Rendah (IPM < 60)
# - Sedang (60 ≤ IPM < 70) 
# - Tinggi (70 ≤ IPM < 80)
# - Sangat tinggi (IPM ≥ 80)

# In[20]:


data['Klasifikasi IPM']=pd.cut(data['IPM'],
                              bins=[0,60,70,80,100],
                              labels=['Rendah','Sedang','Tinggi','Sangat Tinggi'])
data


# In[21]:


data=data.drop(['IPM'], axis = 1)
data.rename(columns={'Klasifikasi IPM':'IPM'}, inplace=True)
data


# In[22]:


fig = data[data.IPM=='Sangat Tinggi'].plot(kind='scatter',x='UHH',y='PPP',color='orange', label='Sangat Tinggi')
data[data.IPM=='Tinggi'].plot(kind='scatter',x='UHH',y='PPP',color='blue', label='Tinggi',ax=fig)
data[data.IPM=='Sedang'].plot(kind='scatter',x='UHH',y='PPP',color='green', label='Sedang', ax=fig)
data[data.IPM=='Rendah'].plot(kind='scatter',x='UHH',y='PPP',color='red', label='Rendah', ax=fig)
fig.set_xlabel("UHH")
fig.set_ylabel("PPP")
fig.set_title("UHH vs PPP")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# Berdasarkan plot di atas dapat diketahui dengan karakteristik UHH dan PPP kita dapat  mengklasifikasikan IPM tetapi ada sedikit garis tipis antara kategori sedang dan tinggi

# In[23]:


fig = data[data.IPM=='Sangat Tinggi'].plot(kind='scatter',x='HLS',y='RLS',color='orange', label='Sangat Tinggi')
data[data.IPM=='Tinggi'].plot(kind='scatter',x='HLS',y='RLS',color='blue', label='Tinggi',ax=fig)
data[data.IPM=='Sedang'].plot(kind='scatter',x='HLS',y='RLS',color='green', label='Sedang', ax=fig)
data[data.IPM=='Rendah'].plot(kind='scatter',x='HLS',y='RLS',color='red', label='Rendah', ax=fig)
fig.set_xlabel("HLS")
fig.set_ylabel("RLS")
fig.set_title("HLS vs RLS")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# Berdasarkan plot di atas dapat diketahui bahwa karakteristik HLS dan RLS dapat membedakan kategori rendah dan sangat tinggi tetapi tidak dengan kategori sedang dan tinggi

# ## Pembentukan Model

# In[24]:


X=data.iloc[:,0:3].values
y=data.iloc[:,4].values


# In[25]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[26]:


#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report

# Import libarary confusion matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.model_selection import train_test_split

# Import libarary Logistic Regression
from sklearn.linear_model import  LogisticRegression

from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

# Import libarary KNN
from sklearn.neighbors import KNeighborsClassifier

# Import libarary Support Vector Machines dan linier Support Vector Machines
from sklearn.svm import SVC, LinearSVC

# Import libarary Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[27]:


#Train and Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# ### KKN

# In[28]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test) 
accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for KNN\n',cm)
print('accuracy_KNN : %.3f' %accuracy)
print('precision_KNN : %.3f' %precision)
print('recall_KNN: %.3f' %recall)
print('f1-score_KNN : %.3f' %f1)


# In[29]:


plt.subplots(figsize=(20,5))
a_index=list(range(1,50))
a=pd.Series()
x=range(1,50)
#x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,50)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train, y_train) 
    prediction=model.predict(X_test)
    a=a.append(pd.Series(accuracy_score(y_test,prediction)))
plt.plot(a_index, a,marker="*")
plt.xticks(x)
plt.show()


# ### Regresi Logistic

# In[30]:


logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
accuracy_lr=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_pred,)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Logistic Regression\n',cm)
print('accuracy_Logistic Regression : %.3f' %accuracy)
print('precision_Logistic Regression : %.3f' %precision)
print('recall_Logistic Regression: %.3f' %recall)
print('f1-score_Logistic Regression : %.3f' %f1)


# ### Naive Bayes

# In[31]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Naive Bayes\n',cm)
print('accuracy_Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1-score_Naive Bayes : %.3f' %f1)


# ### SVM

# In[32]:


linear_svc = LinearSVC(max_iter=4000)
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
accuracy_svc=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for SVC\n',cm)
print('accuracy_SVC: %.3f' %accuracy)
print('precision_SVC: %.3f' %precision)
print('recall_SVC: %.3f' %recall)
print('f1-score_SVC : %.3f' %f1)


# ### Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test) 
accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for DecisionTree\n',cm)
print('accuracy_DecisionTree: %.3f' %accuracy)
print('precision_DecisionTree: %.3f' %precision)
print('recall_DecisionTree: %.3f' %recall)
print('f1-score_DecisionTree : %.3f' %f1)


# ### Random Forest

# In[35]:


# Import Library Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


# In[36]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

accuracy = accuracy_score(y_test,Y_prediction)
precision =precision_score(y_test, Y_prediction,average='micro')
recall =recall_score(y_test, Y_prediction,average='micro')
f1 = f1_score(y_test,Y_prediction,average='micro')
print('accuracy_random_Forest : %.3f' %accuracy)
print('precision_random_Forest : %.3f' %precision)
print('recall_random_Forest : %.3f' %recall)
print('f1-score_random_Forest : %.3f' %f1)


# ## Pemilihan Model Terbaik

# In[37]:


results = pd.DataFrame({
    'Model': [ 'KNN', 
              'Logistic Regression', 
              'Random Forest',
              'Naive Bayes',  
              ' Support Vector Machine', 
              'Decision Tree'],
    'Score': [ acc_knn,
              acc_log, 
              acc_random_forest,
              acc_gaussian,  
              acc_linear_svc,
              acc_decision_tree],
     "Accuracy_score":[accuracy_knn,
                      accuracy_lr,
                      accuracy_rf,
                      accuracy_nb,
                      accuracy_svc,
                      accuracy_dt
                     ]})
result_df = results.sort_values(by='Accuracy_score', ascending=False)
result_df = result_df.reset_index(drop=True)
result_df.head(9)


# In[38]:


plt.subplots(figsize=(12,8))
ax=sns.barplot(x='Model',y="Accuracy_score",data=result_df)
labels = (result_df["Accuracy_score"])
# add result numbers on barchart
for i, v in enumerate(labels):
    ax.text(i, v+1, str(v), horizontalalignment = 'center', size = 15, color = 'black')

