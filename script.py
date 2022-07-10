#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Create Dataframes
admissions = pd.read_csv('admissions.csv')
fatalities = pd.read_csv('fatalities.csv')
smokers = pd.read_csv('smokers.csv')
uk_deaths = pd.read_csv('uk_deaths.csv')

#Display all rows
#pd.set_option('display.max_rows', admissions.shape[0]+1)
#pd.set_option('display.max_rows', fatalities.shape[0]+1)
#pd.set_option('display.max_rows', smokers.shape[0]+1)
#pd.set_option('display.max_rows', uk_deaths.shape[0]+1)

#Drop irrelevant columns
admissions.drop(['ICD10 Code','ICD10 Diagnosis','Metric','Sex'], axis='columns', inplace=True)
fatalities.drop(['ICD10 Code','ICD10 Diagnosis','Metric','Sex'], axis='columns', inplace=True)
smokers.drop(['Method','Sex'], axis='columns', inplace=True)
uk_deaths.drop(['Region'], axis='columns', inplace=True)

#Admissions Dataframes remove last 3 characters eg./14 from Year
for index, row in admissions.iterrows():
    admissions['Year'][index] = admissions['Year'][index][:-3]

#Change year type to int
admissions['Year'] = admissions['Year'].astype(int)

#Order by year and month
admissions = admissions.sort_values('Year')
fatalities = fatalities.sort_values('Year')
smokers = smokers.sort_values('Year')
uk_deaths = uk_deaths.sort_values('Year')

#Remove years that will not be used for analysis, 2006-2010 for anaysis
admissions = admissions.drop(admissions[(admissions['Year'] <  2006) | (admissions['Year'] >  2010)].index)
fatalities = fatalities.drop(fatalities[(fatalities['Year'] <  2006) | (fatalities['Year'] >  2010)].index)
smokers = smokers.drop(smokers[(smokers['Year'] <  2006) | (smokers['Year'] >  2010)].index)
uk_deaths = uk_deaths.drop(uk_deaths[(uk_deaths['Year'] <  2006) | (uk_deaths['Year'] >  2010)].index)

#Drop Nan values
admissions = admissions.dropna()
#fatalities = fatalities.dropna()
#smokers = smokers.dropna()
#uk_deaths = uk_deaths.dropna()

#Find NaN values in DataFrames
ad = admissions.isnull().sum().sum() #0
ft = fatalities.isnull().sum().sum() #0
sm = smokers.isnull().sum().sum() #0
ud = uk_deaths.isnull().sum().sum() #0

#Replace . in value column with 0 and change type to int
#print(admissions[admissions['Value'] == '.'])
admissions.Value.replace(['.'], [0], inplace=True)
admissions['Value'] = admissions['Value'].astype(int)
fatalities.Value.replace(['.'], [0], inplace=True)
fatalities['Value'] = fatalities['Value'].astype(int)

#Create admission stats for each year
admissions06 = 0
admissions07 = 0
admissions08 = 0
admissions09 = 0
admissions10 = 0
for x,y in admissions.iterrows():
	if y['Year'] == 2006:
		admissions06 = admissions06 + admissions.at[x, 'Value'] 
	if y['Year'] == 2007:
		admissions07 = admissions07 + admissions.at[x, 'Value'] 
	if y['Year'] == 2008:
		admissions08 = admissions08 + admissions.at[x, 'Value'] 
	if y['Year'] == 2009:
		admissions09 = admissions09 + admissions.at[x, 'Value'] 
	if y['Year'] == 2010:
		admissions10 = admissions10 + admissions.at[x, 'Value'] 
admissionsarr = [admissions06,admissions07,admissions08,admissions09,admissions10]

#Create fatalities stats for each year
fatalities06 = 0
fatalities07 = 0
fatalities08 = 0
fatalities09 = 0
fatalities10 = 0
for x,y in fatalities.iterrows():
	if y['Year'] == 2006:
		fatalities06 = fatalities06 + fatalities.at[x, 'Value'] 
	if y['Year'] == 2007:
		fatalities07 = fatalities07 + fatalities.at[x, 'Value'] 
	if y['Year'] == 2008:
		fatalities08 = fatalities08 + fatalities.at[x, 'Value'] 
	if y['Year'] == 2009:
		fatalities09 = fatalities09 + fatalities.at[x, 'Value'] 
	if y['Year'] == 2010:
		fatalities10 = fatalities10 + fatalities.at[x, 'Value']
fatalitiesarr = [fatalities06,fatalities07,fatalities08,fatalities09,fatalities10] 

#Create uk death stats for each year
uk_deaths06 = 0
uk_deaths07 = 0
uk_deaths08 = 0
uk_deaths09 = 0
uk_deaths10 = 0
for x,y in uk_deaths.iterrows():
	if y['Year'] == 2006:
		uk_deaths06 = uk_deaths06 + uk_deaths.at[x, 'Deaths'] 
	if y['Year'] == 2007:
		uk_deaths07 = uk_deaths07 + uk_deaths.at[x, 'Deaths'] 
	if y['Year'] == 2008:
		uk_deaths08 = uk_deaths08 + uk_deaths.at[x, 'Deaths'] 
	if y['Year'] == 2009:
		uk_deaths09 = uk_deaths09 + uk_deaths.at[x, 'Deaths'] 
	if y['Year'] == 2010:
		uk_deaths10 = uk_deaths10 + uk_deaths.at[x, 'Deaths']
uk_deathsarr = [uk_deaths06,uk_deaths07,uk_deaths08,uk_deaths09,uk_deaths10] 

#Create smokers stats for each year
smokers = smokers.groupby('Year').sum()
smokers['Total'] = smokers.sum(axis=1)
s = smokers['Total'].values.tolist()

#Create initial plots
plt.plot(s, label = "Smokers")
#plt.xlim([2006,2010])
plt.xlabel('Year(2006-2010)')
plt.ylabel('Smokers(Millions)')
plt.title("Smokers")
plt.legend()
#plt.show()

plt.plot(admissionsarr, label = "Admissions")
#plt.xlim([2006,2010])
plt.xlabel('Year(2006-2010)')
plt.ylabel('Admissions(Millions)')
plt.title("Admissions")
plt.legend()
#plt.show()

plt.plot(fatalitiesarr, label = "Fatalities")
#plt.xlim([2006,2010])
plt.xlabel('Year(2006-2010)')
plt.ylabel('Fatalities(Millions)')
plt.title("Fatalities")
plt.legend()
#plt.show()

plt.plot(uk_deathsarr, label = "UK Deaths")
#plt.xlim([2006,2010])
plt.xlabel('Year(2006-2010)')
plt.ylabel('Deaths')
plt.title("Deaths")
plt.legend()
#plt.show()

#Unique values in Admissions Diagnosis Type
admissionsdiagnosis = admissions['Diagnosis Type'].nunique() #11

#Unique values in Fatalities Diagnosis Type
fatalitiesdiagnosis = fatalities['Diagnosis Type'].nunique() #10

#Group Diagnosis type for each year
admissions = admissions.groupby(['Year','Diagnosis Type'])['Value'].sum()
fatalities = fatalities.groupby(['Year','Diagnosis Type'])['Value'].sum()

#Convert to dataframes
admissions = admissions.to_frame()
fatalities = fatalities.to_frame()
#uk_deaths arleady dataframe
#smokers already dataframe

#Rename columns
admissions=admissions.rename(columns = {'Value':'Total Admissions'})
fatalities=fatalities.rename(columns = {'Value':'No. Fatalities'})
uk_deaths=uk_deaths.rename(columns = {'Deaths':'Total Deaths'})
smokers=smokers.rename(columns = {'Total':'Smokers Admitted'})

#Rearrage ukdeaths dataframe for merge 
uk_deaths = uk_deaths.groupby(["Year"]).sum() #sum months
uk_deaths.drop(['Month'], axis='columns', inplace=True) #drop months
uk_deaths.insert(0, 'Diagnosis Type', "All Deaths")
#uk_deaths.insert(1, 'Total Admissions', 0)

#Merge Datasets
merge1 = pd.merge(admissions, fatalities, on=['Year','Diagnosis Type'], how='inner')
merge1 = merge1.sort_values('Year')

merge2 = pd.merge(uk_deaths, smokers, on=['Year'], how='left')
merge2 = merge2.sort_values('Year')

merge = pd.merge(merge1, merge2, on=['Year','Diagnosis Type'], how='outer')
merge = merge.sort_values('Year')

#Fix data types
merge = merge.fillna(0)
merge.loc[merge['Diagnosis Type'] == 'All cancers', 'Diagnosis Type'] = 1
merge.loc[merge['Diagnosis Type'] == 'All circulatory diseases', 'Diagnosis Type'] = 2
merge.loc[merge['Diagnosis Type'] == 'All diseases of the digestive system', 'Diagnosis Type'] = 3
merge.loc[merge['Diagnosis Type'] == 'All respiratory diseases', 'Diagnosis Type'] = 4
merge.loc[merge['Diagnosis Type'] == 'Cancers which can be caused by smoking', 'Diagnosis Type'] = 5
merge.loc[merge['Diagnosis Type'] == 'Circulatory diseases which can be caused by smoking', 'Diagnosis Type'] = 6
merge.loc[merge['Diagnosis Type'] == 'Digestive diseases which can be caused caused by smoking', 'Diagnosis Type'] = 7
merge.loc[merge['Diagnosis Type'] == 'Respiratory diseases which can be caused by smoking', 'Diagnosis Type'] = 8
merge.loc[merge['Diagnosis Type'] == 'All Deaths', 'Diagnosis Type'] = 9
merge['Diagnosis Type'] = merge['Diagnosis Type'].astype(float)

#Edit columns
merge = merge.reset_index()
merge.columns = ['Year','Diagnosis Type','Total Admissions','No. Fatalities','Total Deaths','16 and Over','16-24','25-34','35-49','50-59','60 and Over','Smokers Admitted']

#One-hot encoding
dumyvar = pd.get_dummies(merge,drop_first=True)
dumyvar = pd.concat([merge,dumyvar],axis=1)
dumyvar.drop(merge,axis=1)
dumyvar.dropna()

#Feature Importance
x = merge.drop(['Diagnosis Type'],axis=1)
y = merge['Diagnosis Type']
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
kbest = SelectKBest(score_func = chi2,k='all')
ordered_features = kbest.fit(x,y)
df_scores = pd.DataFrame(ordered_features.scores_,columns=['Diagnosis Type'])
df_columns = pd.DataFrame(x.columns,columns=['Feature_name'])
feature_rank = pd.concat([df_scores,df_columns],axis=1)
feature_rank = feature_rank.nlargest(11,'Diagnosis Type')
#print(feature_rank)
model = ExtraTreesClassifier()
model.fit(x,y)
ranked_features = pd.Series(model.feature_importances_,index=x.columns)
ranked_features = ranked_features.nlargest(11).plot(kind='bar')
#plt.show()


#Information gain
mu_ifo = mutual_info_classif(x,y)
mu_data = pd.Series(mu_ifo,index=x.columns)
mu_data = mu_data.sort_values(ascending=False)
#print(mu_data)



#Modelling
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
def classify(model,x,y):
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
	model.fit(x_train,y_train)
	print('Accuracy is : ', model.score(x_test,y_test)*100)
	score = cross_val_score(model,x,y,cv=5)
	print('Cross validation accuracy: ', np.mean(score)*100)

#KNN
knn = KNeighborsClassifier(n_neighbors=15)
#classify(knn,x,y)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

knn.fit(x_train, y_train)
knnpredict = knn.predict(x_test)
#print(classification_report(y_test,knnpredict))

knaccuracy = knn.score(x,y)
knnaccuracy = accuracy_score(y_test,knnpredict)
#print(knnaccuracy)

error = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Values')
plt.xlabel('k')
plt.ylabel('Error')
#plt.show()


#Linear Regression
linreg = linear_model.LinearRegression()
plt.figure(figsize=(12,10))
for f in range(0,10):
    xi_test = x_test[:,f]
    xi_train = x_train[:,f]
    xi_test = xi_test[:,np.newaxis]
    xi_train = xi_train[:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y = linreg.predict(xi_test)
    
    plt.subplot(5,2,f+1)
    plt.scatter(xi_test,y_test,color='k')
    plt.plot(xi_test,y,color='b',linewidth=1.5)
#plt.show()
linreg.fit(x_train, y_train)
linpred = linreg.predict(x_test)
linpredict = linreg.score(x_test,linpred)
linaccuracy = linreg.score(x_test,y_test)
#print(linaccuracy)

#Confusion Matrix
from sklearn.svm import SVC
clf = SVC(kernel='linear',C=1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
#plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(merge.corr(),annot=True,cmap="BuPu")
#plt.show()

#Analysis
accuracyscores = {"Knn: ": knnaccuracy,"Linear Regression:": linaccuracy}
a = np.arange(len(accuracyscores))
p = plt.subplot(111)
p.bar(a,accuracyscores.values(), width=0.2, color='b')
plt.xticks(a,accuracyscores.keys())
plt.title("Accuracy of Knn and Linear Regression")
#plt.show()