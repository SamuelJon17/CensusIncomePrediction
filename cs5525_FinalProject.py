
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
sns.set(style="darkgrid")

## Transformations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

## Feature Reduction Analyis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

## Analaysis
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
import scipy.stats 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import auc

## Models
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import statsmodels.api as sm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

## State for all randomness
randomState = 17

#####################################################
# Data Cleaning
#####################################################
## Smaller data set with ~10 features

colNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dfTrain = pd.read_csv(os.path.join(os.getcwd(),"subset","adult.data"), index_col=False, names = colNames)
dfTest = pd.read_csv(os.path.join(os.getcwd(),"subset","adult.test"), index_col=False, names = colNames)
df = pd.concat([dfTrain,dfTest.iloc[1::]])

## Don't require fnlwgt for analysis
df.drop(columns=["fnlwgt"], inplace = True)

# Read in age as int type
df['age'] = df.age.astype(int)

## Map target dataset to binary 
df.income = df.income.map({' <=50K': 0, ' <=50K.': 0, ' >50K':1, ' >50K.':1})

# Identify null
print(f"Results of Na in the dataset:\n{df.isna().sum()}")

# Identify any filler entries that act as null
print("--------------------------------------")
print("Identify any NaN fillers, '?', and convert their entry to np.nan.\n\
Also, want to determine the percentage of data fillers make up to determine\n\
proceeding steps.\n")
for i in df.columns:
    res = len(df[df[i] == ' ?'])
    if res != 0:
        print(f"{i}: {res} records with '?' as entry")
        print(f"This makes up {100*res/len(df):0.2f}% of the dataset\n")
        df.loc[df[i]==' ?', i] = np.nan
missData = 100*len(df[df.workclass.isna() | df.occupation.isna() |  df["native-country"].isna()])/len(df)
print(f"If I were to handle missing data by removing those entries all together that would make up {missData:0.2f}% of the data removed")
print(f"Because I believe I can still obtain a reliable model while still removing {missData:0.2f}% of data,\n\
I will proceed with removing those entries.")
df.dropna(inplace = True)

# Strip data on left side since there seems to be an additional space between entry
catData = set(df.columns) - set(df.describe().columns)
numData= list(set(df.columns)-catData - {"income"})
for i in catData:
    df[i] = df[i].str.lstrip()
    

#####################################################
# Visual Exploratory Analysis
#####################################################

def getGroupPlot(df, columns=["workclass","income"], value = "income"):
    data = df.groupby(columns)[value].count().unstack()
    data = data.div(data.apply(sum,1),0)
    data.sort_values(by = 1, inplace = True)
    data.plot(kind="bar", 
              figsize=(15, 10), 
              title= (columns[0] + " vs. " + columns[1]).title(),
              ylabel = 'Percentage (%)')
    plt.legend(["<=50K", ">50K"])
    plt.tight_layout()
    plt.show()
    return data

# Bin hours per week into four categories for visualization purposes only
df.loc[df['hours-per-week'].between(0, 40, 'left'), 'hoursPerWeek'] = '<40'
df.loc[df['hours-per-week'].between(40, 60, 'left'), 'hoursPerWeek'] = '<60'
df.loc[df['hours-per-week'].between(60, 80, 'left'), 'hoursPerWeek'] = '<80'
df.loc[df['hours-per-week'].between(80, 100, 'both'), 'hoursPerWeek'] = '>=80'
catData = catData|{'hoursPerWeek'}

groupbyAll = {}
for i in catData:
    groupbyAll[i] = (getGroupPlot(df, columns=[i,"income"],value="income"))

# Drop hoursPerWeek & education for the remainder of the modeling since education-num gives the same information
# as education and hoursPerWeek were only used for visualization
df.drop(columns=["hoursPerWeek", "education"], inplace = True)
catData = catData - {'hoursPerWeek', 'education'}

#####################################################
# Data Transformation
#####################################################

# Split data prior to transformations
X_train, X_test, y_train, y_test = train_test_split(df[list(set(df.columns)-{"income"})], df['income'], test_size=0.2, random_state=randomState)

X_train[numData].plot(kind = 'box', 
            figsize = (15,10),
            title="Numerical Data Before Transformation",
            xlabel = "Features",
            ylabel="Numerical Value")


# Skew value <-1 or >1 is highly skewed
X_train[numData].skew().sort_values(ascending=False)

## Visualize Skewness
fig = plt.figure(figsize = (15,20));
for i, feature in enumerate(X_train[numData]):
    ax = fig.add_subplot(3, 2, i+1)
    ax.hist(X_train[feature], bins = 50)
    ax.set_title(f"{feature.title()} Feature Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Feature Count")
    ax.set_ylim((0, 3000))
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels([0, 1000, 2000, ">3000"])
fig.suptitle("Distributions of Numerical Features")
fig.tight_layout()
plt.show()

# Adjust for skewness on capital gain and capital loss
powerSkew = PowerTransformer()
fig = plt.figure(figsize=(15,10))

j = 1
for i in ["capital-gain","capital-loss"]:
    ax = fig.add_subplot(2, 2, j)
    ax.hist(X_train[i], bins = 50)
    ax.set_title(f"Orginal Distribution for {i}")
    ax.set_ylim((0, 3000))
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels([0, 1000, 2000, ">3000"])
    j+=2

powerSkew = PowerTransformer()
X_train[['capital-gain','capital-loss']]=powerSkew.fit_transform(X_train[['capital-gain','capital-loss']])
X_test[['capital-gain','capital-loss']] = powerSkew.transform(X_test[['capital-gain','capital-loss']])

j=2
for i in ["capital-gain","capital-loss"]: 
    ax = fig.add_subplot(2, 2, j)
    ax.hist(X_train[i], bins = 50)
    ax.set_title(f"Power Transform for {i}")
    j += 2

plt.show()
X_train[numData].skew().sort_values(ascending=False)

X_train[numData].plot(kind = 'box', 
            figsize = (15,10),
            title="Numerical Data After Skew Transformation",
            xlabel = "Features",
            ylabel="Numerical Value")

## Because I know the capital loss and capital-gain do not follw a normal distribution still, I am
## choosing to normalize all the numerical data rather than standardize
normalize = MinMaxScaler()
X_train[numData] = normalize.fit_transform(X_train[numData])
X_test[numData] = normalize.transform(X_test[numData])
X_train.head(5)

# Used to identify if normalization is required
X_train[numData].plot(kind = 'box', 
                figsize = (15,10),
                title="Numerical Data After Normalization",
                xlabel = "Features",
                ylabel="Numerical Value")

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_trainCat = pd.DataFrame(encoder.fit_transform(X_train[catData]))
X_testCat = pd.DataFrame(encoder.transform(X_test[catData]))

# Grab index location of categorical data to pass get_feature_names
catIdx = [X_train.columns.get_loc(col) for col in catData]

# Adding column names to the encoded data set
X_trainCat.columns = encoder.get_feature_names(X_train.columns.values[catIdx].tolist())
X_testCat.columns = encoder.get_feature_names(X_train.columns.values[catIdx].tolist())

# One-hot encoding removed index; put it back
X_trainCat.index = X_train.index
X_testCat.index = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
X_trainNum = X_train.drop(catData, axis=1)
X_testNum = X_test.drop(catData, axis=1)

# Add one-hot encoded columns to numerical features
X_train2 = pd.concat([X_trainNum, X_trainCat], axis=1)
X_test2 = pd.concat([X_testNum, X_testCat], axis=1)

#####################################################
# Dimensionality Reduction
#####################################################

## SVD
svdThresh = 0.85
svdVar = []
for i in range(1,len(X_train2.columns)+1):
    svdVar.append({"n_components":i, "explainedVar":sum(TruncatedSVD(n_components = i).fit(X_train2).explained_variance_ratio_)})
svdVar = pd.DataFrame(svdVar)

plt.figure(figsize=(10,10))
plt.plot(svdVar['n_components'], svdVar['explainedVar'], label = 'Explained Variance')
plt.axhline(y = svdThresh, color = 'r', linestyle = ':')
plt.legend()
plt.xlabel("SVD Components")
plt.ylabel("Explained Variance")
plt.title("Number of SVD Components vs. Explained Varaince")
plt.show()

print(f"SVD suggest that there should be at least {svdVar[svdVar['explainedVar']>=svdThresh].iloc[0]['n_components']:0.0f} of {len(svdVar)} SVD components\n\
to provide {svdThresh*100}% variance")

## PCA
pcaThresh = 0.85
pcaVar = []
for i in range(1,len(X_train2.columns)+1):
    pcaVar.append({"n_components":i, "explainedVar":sum(PCA(n_components = i).fit(X_train2).explained_variance_ratio_)})
pcaVar = pd.DataFrame(pcaVar)

plt.figure(figsize=(10,10))
plt.plot(pcaVar['n_components'], pcaVar['explainedVar'], label = 'Explained Variance')
plt.axhline(y = pcaThresh, color = 'r', linestyle = ':')
plt.legend()
plt.xlabel("PCA Components")
plt.ylabel("Explained Variance")
plt.title("Number of PCA Components vs. Explained Varaince")
plt.show()

print(f"PCA suggest that there should be at least {pcaVar[pcaVar['explainedVar']>=pcaThresh].iloc[0]['n_components']:0.0f} of {len(pcaVar)} PCA components\n\
to provide {pcaThresh*100}% variance")

## Random Forest Analysis (RFA)
thresh = [0.001,0.005,0.0075, 0.01,0.05,0.1]
threshFeatures = {}
rfAnalysis = RandomForestClassifier(random_state=randomState).fit(X_train2, y_train)
X_train2_FeatureImportance = pd.DataFrame(rfAnalysis.feature_importances_, X_train2.columns).round(4).sort_values(by=[0], ascending=False)
for i in thresh:
    print("===========================")
    threshCol = list(X_train2_FeatureImportance[X_train2_FeatureImportance[0]>i].index)
    threshFeatures[str(i)] = threshCol
    logThresh = LogisticRegression(random_state=randomState)
    logThresh = logThresh.fit(X_train2[threshCol], y_train)
    print(f"\nNumber of features selected: {len(threshCol)}")
    print(f"\nFeatures include {threshCol}")
    print(f"\nRandom Forest Top {i} Threshold, Accuracy for Logistic Regression: {accuracy_score(y_test, logThresh.predict(X_test2[threshCol])):0.3f}")

print("Because PCA, SVD suggest 18-20 features of 87 to cover 85% variance and the Random Forest Analysis\n\
suggest 14-28 features with feature importance threshold of 0.005-0.01, I will select threshold 0.0075.\n\
At 0.0075, this meets the PCA/SVD variance coverage and minimizes the feature space significantly without\n\
diminishing the accuracy of a logistic regression. Although 5-14 features still produces a great accuracy\n\
according to PCA/SVD, reducing the feature space this much would reduce variance coverage.")

X_train2_FeatureImportance = X_train2_FeatureImportance[X_train2_FeatureImportance[0]>0.0075].sort_values(by=[0], ascending=True)
plt.figure(figsize=(15,10))
plt.barh(range(len(X_train2_FeatureImportance)), X_train2_FeatureImportance[0]) 
plt.yticks(range(len(X_train2_FeatureImportance)), X_train2_FeatureImportance.index) 
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.axvline(x = 0.0075, color = 'r', linestyle = ':')
plt.title(f"Top {len(X_train2_FeatureImportance)} Features Based On 0.0075 Feature Importance Through Random Forest Analysis")
plt.show()

X_train3 = X_train2[list(X_train2_FeatureImportance.index)]
X_test3 = X_test2[list(X_train2_FeatureImportance.index)]

#####################################################
# Covariance and Correlation Matrix
#####################################################

## Cov
plt.figure(figsize=(10,10))
sns.heatmap(pd.concat([X_train3, y_train], axis=1).cov(), annot=False, cmap="seismic")
plt.title(f"Covariance Matrix for All {len(X_train3.columns)} Features w/ Income",  fontsize=20)
plt.show()

## Corr
plt.figure(figsize=(10,10))
sns.heatmap(pd.concat([X_train3, y_train], axis=1).corr(method='pearson'), annot=False, cmap="seismic")
plt.title(f"Correlation Matrix for All {len(X_train3.columns)} Features w/ Income",  fontsize=20)
plt.show()


#####################################################
# T-test and F-test
#####################################################
tTest = []
fTest = []
alpha = 0.05

for i in range(len(X_train3.columns)):
    for k in range(i+1, len(X_train3.columns)):
        iCol = X_train3.columns[i]
        kCol = X_train3.columns[k]
        
        ## f test
        f = np.var(X_train3[iCol], ddof=1)/np.var(X_train3[kCol], ddof=1)
        f_pvalue = 1-scipy.stats.f.cdf(f, X_train3[iCol].size-1, X_train3[kCol].size-1)
        if f_pvalue <= alpha:
            fTest.append({"feature1":iCol, "feature2":kCol, "f-test":f, "p-value":f_pvalue})
        
        ## t test
        tResults = ttest_ind(X_train3[iCol], X_train3[kCol])
        if tResults.pvalue <= alpha:
            tTest.append({"feature1":iCol, "feature2":kCol, "t-test":tResults.statistic, "p-value":tResults.pvalue})
tTest = pd.DataFrame(tTest).round(3)
fTest = pd.DataFrame(fTest).round(2)
tTest.sort_values(by=["t-test"],ascending=False)
fTest.sort_values(by=["f-test"],ascending=False)

#####################################################
# Collinearity Assessment w/ Variance Inflation Factor
#####################################################

vif_scores = pd.DataFrame() 
vif_scores["Attribute"] = X_train3.columns 
vif_scores["vifScores"] = [variance_inflation_factor(X_train3.values, i) for i in range(len(X_train3.columns))] 

print("It is known that colinearity will exist with binary one hot encoded features such as male and female.\n\
However upon reseearch, it suggest this analysis can be ignored for said features and instead ensure\n\
when modeling a regression to exclude the intercept. Similar information can be made for husband/wife.\n\
It might make sense to remove Married Civilian Spouse since Married-spouse-absent and Married-AF-spouse\n\
have been removed earlier and can infer this info from relationship (husband/wife). However for the\n\
remainder of the modeling, will leave feature space as is.")

vif_scores2 = pd.DataFrame() 
tempCol = set(X_train3.columns) - {"marital-status_Married-civ-spouse"}
vif_scores2["Attribute"] = list(tempCol)
vif_scores2["vifScores"] = [variance_inflation_factor(X_train3[list(tempCol)].values, i) for i in range(len(tempCol))]
vif_scores.merge(vif_scores2, on="Attribute", how="left", suffixes=("_original","_updated")).round(2).sort_values(by=['vifScores_original'], ascending =False)

#####################################################
# Backward Stepwise Regression w/ 0.05 threshold
#####################################################

def backwardStepwise(X_train, y_train, X_test, y_test, thresh = 0.05):
    features=list(X_train.columns)
    while True:
        changed=False
        model = sm.Logit(y_train, sm.add_constant(X_train[features])).fit()
        pvalues = model.pvalues.iloc[1:]
        if pvalues.max() > thresh:
            changed=True
            features.remove(pvalues.idxmax())
            print(f'Feature "{pvalues.idxmax()}" dropped (p-value {pvalues.max():.2f} > thresh {thresh})')
        if not changed:
            break
    print(f"\nFeatures kept by backwards stepwise regression are: {', '.join(features)}")
    return features, accuracy_score(y_test, model.predict(sm.add_constant(X_test[features])).round())


features, accuracy = backwardStepwise(X_train3, y_train, X_test3, y_test, thresh=0.05)
print(f"\nAccuracy: {accuracy:0.3f}\n")
print("\nBecause there are no real accuraccy gains after stepwise regression, choosing to leave feature space as is.")

#####################################################
# Final Logistic Regression using RFA features
#####################################################
log1Stats = sm.Logit(y_train, sm.add_constant(X_train3)).fit()
print(log1Stats.summary())

log1SciKit = LogisticRegression(random_state=randomState, fit_intercept=False)
log1SciKit = log1SciKit.fit(X_train3, y_train)

print(f"Accuracy for StatsModel: {accuracy_score(y_test, log1Stats.predict(sm.add_constant(X_test3)).round()):0.3f}")
print(f"Accuracy for SciKit: {accuracy_score(y_test, log1SciKit.predict(X_test3)):0.3f}")

#####################################################
# Base Model Evaluation
#####################################################

def classifierEval(clf, X_train, y_train, X_test, y_test): 
    results = {}
    
    start = time()
    clf = clf.fit(X_train, y_train)
    results['train_time'] = time() - start
    
    start = time()
    pred = clf.predict(X_test)
    results['pred_time'] = time() - start
    
    confusion = confusion_matrix(y_test,pred).ravel()
    tn, fp, fn, tp = confusion 
    
    results['confusion'] = confusion
    results['accuracy'] =  accuracy_score(y_test,pred)
    results['precision'] = precision_score(y_test, pred)
    results['recall'] = recall_score(y_test,pred)
    results['f_measure'] = f1_score(y_test,pred)
    results['specificity'] = tn/(tn+fp)
    
    return clf, results

clf_Log = log1SciKit
clf_DTree = tree.DecisionTreeClassifier(random_state = randomState)
clf_KNN = KNeighborsClassifier(n_neighbors=3)
clf_SVM = svm.SVC(random_state = randomState)
clf_Bayes = GaussianNB()

# Collect results on the learners
results = {}
for clf in [clf_Log, clf_DTree, clf_KNN, clf_SVM, clf_Bayes]:
    print(clf.__class__.__name__)
    clf, results[clf.__class__.__name__] = classifierEval(clf, X_train3, y_train, X_test3, y_test)

## Base Model Metrics
colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
baseResults = pd.DataFrame(results)
baseResults = baseResults.append(pd.DataFrame([colors], index=["colors"], columns=baseResults.columns))

fig = plt.figure(figsize=(20,20))
for j, label in enumerate(["train_time","pred_time","accuracy","precision","recall","f_measure","specificity"]):
    ax = fig.add_subplot(4, 2, j+1)
    temp = baseResults.sort_values(by=[label], axis=1)
    ax.bar(temp.loc[label].index, temp.loc[label].values, color = list(temp.loc["colors"]))
    ax.set_title(f"{label} vs. Models", fontsize=15)
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
plt.suptitle("Performance Metrics for Base Classification Models", fontsize = 16, y = 1)
plt.tight_layout()
plt.show()

baseResults.loc[list(set(baseResults.index) - {"colors","confusion"})].T.round(2)

## ROC and AUC for Base Model
fig = plt.figure(figsize=(10,10))
for i, clf in enumerate([clf_Log, clf_DTree, clf_KNN, clf_SVM, clf_Bayes]):
        if clf.__class__.__name__ == "SVC":
            fpr, tpr, _ = roc_curve(y_test, clf.decision_function(X_test3))
        else:
            fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test3)[:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i],lw=2, label=f'{clf.__class__.__name__} auc = {roc_auc:.2f}')
    
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Base Classification Models')
plt.legend()
plt.show()

## Confusion Matrix
fig = plt.figure(figsize=(20,20))
j=0
for index, value in baseResults.loc['confusion'].iteritems():
    ax = fig.add_subplot(3, 2, j+1)
    sns.heatmap(value.reshape(2, -1), annot=True, fmt="g", ax=ax, cmap="magma")
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_xticklabels(["<=50K",">50K"])
    ax.set_ylabel("True Label", fontsize=15)
    ax.set_yticklabels(["<=50K",">50K"], rotation = 0)
    ax.set_title(f"Confusion Matrix for {index}", fontsize=15)
    j+=1
    
plt.tight_layout()
plt.show()

#####################################################
# Optimized Model Performance with GridSearchCV
# GridSearchCV uses default scoring
#####################################################
def classifierGrid(clf, param_dict, X_train, y_train, X_test, y_test):
    gridCLF =  GridSearchCV(clf,
                      param_grid=param_dict,
                      cv=5)
    gridCLF, results= classifierEval(gridCLF, X_train, y_train, X_test, y_test )
    results["best_params"] = gridCLF.best_params_
    results["score_mean"] = gridCLF.cv_results_['mean_test_score'].mean()
    results["score_best"] = gridCLF.best_score_
    return gridCLF, results

    
clf_DTree_Grid = tree.DecisionTreeClassifier(random_state = randomState)
clf_KNN_Grid = KNeighborsClassifier()
clf_SVMKernel_Grid = svm.SVC(random_state = randomState)

param_clf_DTree_Grid = {"criterion":["gini", "entropy"],
              "max_depth":range(1,15), 
              "min_samples_split":range(2,10), 
              "min_samples_leaf":range(1,5)
             }
param_clf_KNN_Grid = {"n_neighbors":range(1,50,2)}

param_clf_SVMKernel = {"kernel":["poly","linear","rbf","sigmoid"]}

## Running Grid for SVM - Kernel only and will optimize specific kernel next
    ## Note, I understand that this is a greedy approach but saves time for training/testing since SVM takes a while 
clfKey = ['clf_DTree_Grid','clf_KNN_Grid','clf_SVMKernel_Grid']
clfs = dict(zip(clfKey,[clf_DTree_Grid,clf_KNN_Grid,clf_SVMKernel_Grid]))
params = dict(zip(clfKey,[param_clf_DTree_Grid,param_clf_KNN_Grid,param_clf_SVMKernel]))

resultsOpt = {}
for key, clf in clfs.items():
    print("=================")
    print(clf.__class__.__name__)
    print(key)
    start = time()
    clf, resultsOpt[key] = classifierGrid(clf,params[key], X_train3, y_train, X_test3, y_test)
    print(time()-start)
    
####### Best kernel from prior
clf_SVMLinear_Grid = svm.SVC(kernel="linear", random_state = randomState)
param_clf_SVMLinear_Grid = {'C': [0.1, 1, 10, 100]}

start = time()
clf_SVMLinear_Grid, resultsOpt["clf_SVMLinear_Grid"] = classifierGrid(clf_SVMLinear_Grid,param_clf_SVMLinear_Grid, X_train3, y_train, X_test3, y_test)
print(time()-start)

for key, value in resultsOpt.items():
    print(f"Best parameters for {key} are: {resultsOpt[key]['best_params']}")

resultsComb = pd.DataFrame(resultsOpt).join(pd.DataFrame(results)).rename(columns=
                {"LogisticRegression":"clf_Log_Base",
                    "DecisionTreeClassifier":"clf_DTree_Base",
                    "KNeighborsClassifier":"clf_KNN_Base",
                    "SVC":"clf_SVM_Base",
                    "GaussianNB":"clf_GaussianNB_Base"
                }).drop(columns=["clf_SVMKernel_Grid"])
resultsComb = resultsComb.append(pd.DataFrame([['#4f2bdb','#db4f2b','#2bb7db', '#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']], index=["colors"], columns=resultsComb.columns))

resultsComb.loc[["train_time","pred_time","accuracy","precision","recall","f_measure","specificity"]].T

## All Model (Base and Optimized Metrics)
fig = plt.figure(figsize=(20,20))
for j, label in enumerate(["train_time","pred_time","accuracy","precision","recall","f_measure","specificity"]):
    ax = fig.add_subplot(5, 2, j+1)
    temp = resultsComb.sort_values(by=[label], axis=1)
    ax.bar(temp.loc[label].index, temp.loc[label].values, color = list(temp.loc["colors"]))
    ax.set_title(f"{label} vs. Models", fontsize=15)
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
plt.suptitle("Performance Metrics for All Classification Models", fontsize = 16, y = 1)
plt.tight_layout()
plt.show()

## Only optimized model metrics
fig = plt.figure(figsize=(20,20))
cols = list(set(resultsComb.columns)-{"clf_DTree_Base","clf_KNN_Base","clf_SVM_Base"})
for j, label in enumerate(["train_time","pred_time","accuracy","precision","recall","f_measure","specificity"]):
    ax = fig.add_subplot(5, 2, j+1)
    temp = resultsComb[cols].sort_values(by=[label], axis=1)
    ax.bar(temp.loc[label].index, temp.loc[label].values, color = list(temp.loc["colors"]))
    ax.set_title(f"{label} vs. Models", fontsize=15)
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
plt.suptitle("Performance Metrics for Optimized Classification Models", fontsize = 16, y = 1)
plt.tight_layout()
plt.show()

## Only optimized model confusion matrix
fig = plt.figure(figsize=(20,20))
j=0
for index, value in resultsComb[cols].loc['confusion'].iteritems():
    ax = fig.add_subplot(3, 2, j+1)
    sns.heatmap(value.reshape(2, -1), annot=True, fmt="g", ax=ax, cmap="magma")
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_xticklabels(["<=50K",">50K"])
    ax.set_ylabel("True Label", fontsize=15)
    ax.set_yticklabels(["<=50K",">50K"], rotation = 0)
    ax.set_title(f"Confusion Matrix for {index}", fontsize=15)
    j+=1
plt.tight_layout()
plt.show()

## Only optimized model ROC and AUC
fig = plt.figure(figsize=(10,10))
labels = ["clf_Log_Base", "clf_GaussianNB_Base", "clf_DTree_Grid", "clf_KNN_Grid", "clf_SVMLinear_Grid"]
for i, clf in enumerate([clf_Log, clf_Bayes, clf_DTree_Grid, clf_KNN_Grid, clf_SVMLinear_Grid]):
        if clf.__class__.__name__ == "SVC":
            fpr, tpr, _ = roc_curve(y_test, clf.decision_function(X_test3))
        else:
            fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test3)[:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=resultsComb[labels].loc["colors"].values[i],lw=2, label=f'{labels[i]} auc = {roc_auc:.2f}')
    
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Optimized Classification Models')
plt.legend()
plt.show()