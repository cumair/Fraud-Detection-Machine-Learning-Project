#!/usr/bin/env python
# coding: utf-8

# ### Final Project Notebook
# # Fraud Detection in Auto Insurance Claims
# ### Analysis Using Machine Learning Techniques
# ### Umair Chaanda

# The main goal of this project is the implementation of various data mining and machine learning techniques and their applications. The exploration of multiple data analysis tasks on the targeted data, including both supervised knowledge discovery (predictive modeling) as well as unsupervised knowledge discovery for exploratory data analysis. The basic objective is to create a best predictive model for classifying fraud, whether a claimant is likely to commit fraud in auto insurance claims. This is a pure classification task which also includes the misclassification costs. Accurate prediction of fraud in claims will lead to a number of benefits described above to several stakeholders.

# ## Import Packages

# In[1]:


from numpy import *
import numpy as np                                          # for carrying out efficient computations
from numpy import linalg as la
import pylab as pl
import pandas as pd                                         # for reading and writing spreadsheets
import pdb
import matplotlib.pyplot as plt                             # for displaying plots
import seaborn as sns                                       # for visualization of data
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split        # for splitting data into train and test sets
from sklearn import preprocessing                           # for min-max normalization

from sklearn import neighbors                               # KNN Classifier
from sklearn.neighbors import KNeighborsClassifier          # KNeighborsClassifier
from sklearn import tree                                    # Decision tree Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Linear Discriminant Analysis (LDA)
from sklearn import naive_bayes                             # Naive bayes classifier
from sklearn.neighbors import NearestCentroid               # Rocchio Classifier
from sklearn.ensemble import RandomForestClassifier         # Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier             # Ada Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier     # Gradient Boosting Classifier
from sklearn.ensemble import BaggingClassifier              # Bagging Classifier
from sklearn.svm import SVC                                 # SVC Classifier (Support Vector Machines)
from sklearn.linear_model import LogisticRegression         # Logistic Regression Classifier

from sklearn.metrics import classification_report           # for classification report
from sklearn.metrics import confusion_matrix                # for confustion matrix

from sklearn.model_selection import cross_val_score, KFold  # k-fold cross validation

from sklearn.linear_model import LinearRegression           # standard multiple regression
from sklearn.linear_model import Lasso                      # L1 regularized regression
from sklearn.linear_model import Ridge                      # L2 regularized regression
from sklearn.linear_model import ElasticNet                 # combines the Lasso and Ridge
from sklearn.linear_model import SGDRegressor               # linear regression where SGD optimization is used to learn the model

from sklearn.cluster import KMeans                          # KMeans clustering function from sklearn.cluster
from sklearn import metrics                                 # Measure model performance
from sklearn.metrics import completeness_score, homogeneity_score

from sklearn import decomposition                           # PCA

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from scipy.stats import sem

from sklearn.model_selection import GridSearchCV            # To optimize model parameters


# In[2]:


import warnings
warnings.filterwarnings('ignore')                           # ignore displaying the warnings


# In[3]:


get_ipython().run_line_magic('pylab', 'inline')


# ## Read Data File
# The data used in this project is publicly-available and acquired from Kaggle. It contains information about historic claims and policy information. It has 39 columns including numerical, categorical and other types of variables with 1000 rows.

# In[4]:


claims = pd.read_csv("insurance_claims.csv", na_values=["?"])
                                       # missing values in csv file are listed as ?
                                       # Replace it with NaN in dataframe
print('Rows,', 'Columns')
print(claims.shape)                    # print number of rows and columns in data
claims.head()                          # see the first five rows in data


# In[5]:


claims.info()


# ## Data Cleaning, Exploration/Visualization and Preprocessing
# There are several data cleaning, data exploration and preprocessing steps will be performed as part of the project.

# ### General information about the data and Examine and handle the missing data

# In[6]:


# display general information about the data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':claims.dtypes,
              'Levels':[claims[x].unique() for x in claims.columns],
              'Null_Count':claims.isna().sum(),
              'Number_Unique_Values':claims.nunique()
             })


# ### Basic Statistics of Features:
# Let's explore the general characteristics of the data as a whole: examine the means, standard deviations, and other statistics associated with the numerical attributes.

# In[7]:


# summary of the distribution of features
claims.describe(include="all").T


# After reading the csv dataset into a Dataframe in python, the first step was to look at the general information about the data such as feature names and their types, missing values, number of unique values and the categorical levels for each feature. The second step was to look at the basic statistics of features to explore the general characteristics of the data as a whole: examine the means, standard deviations, and other statistics associated with the numerical attributes.

# ### Discretization - Convert numerical attributes into categorical attributes
# Data discretization is defined as a process of converting continuous data attribute values into a finite set of intervals and associating with each interval some specific data value.

# In[8]:


# make a copy of original dataframe
claims_clean = claims.copy()


# In[9]:


# incident_hour_of_the_day has 24 unique values
# Discretize the incident_hour_of_the_day attribute into 3 categories (corresponding to 'Morning', 'Afternoon', 'Evening')
claims_clean['incident_part_of_the_day'] = pd.cut(claims_clean.incident_hour_of_the_day,bins=[0, 11, 17, 23],
                                                  labels=['Morning', 'Afternoon', 'Evening'], include_lowest=True)


# The attribute (incident_hour_of_the_day) had 24 uniques values and it was not suitable to treat this variable as numerical so we discretized the incident_hour_of_the_day attribute into 3 categories corresponding to 'Morning', 'Afternoon', 'Evening' (bins=[0, 11, 17, 23]) using pandas pd.cut function.

# ### Convert Some Variables Types: 
# In many situations, we need to convert variables from one type into another. Type conversion is a method of changing features from one data type to another. An example of typecasting is converting an integer to a string.

# In[10]:


# convert these variables from int64 into object
int_list = ['policy_deductable', 'number_of_vehicles_involved', 
            'bodily_injuries', 'witnesses']

# policy_deductable, number_of_vehicles_involved, bodily_injuries, witnesses
claims_clean[int_list] = claims_clean[int_list].astype(str)

# convert policy_annual_premium from float to int64
claims_clean['policy_annual_premium'] = claims_clean['policy_annual_premium'].astype(int64)


# The features ('policy_deductable', 'number_of_vehicles_involved', 'bodily_injuries', and 'witnesses')  had very few unique values and they are actually categorical in nature so we converted the data type of these features from int64 into objects. We also converted the feature (‘policy_annual_premium’) from float to int64 for efficient  computation.

# ### Handling Missing Data and Dropping some Features:

# In[11]:


# drop variables which are not going to be used in our analysis based on above information
drop_col = ['policy_number', 'policy_bind_date', 'insured_zip', 
            'incident_date', 'collision_type', 'incident_location', 
            'property_damage', 'police_report_available', 'auto_model',
            'incident_hour_of_the_day']
claims_clean.drop(columns=drop_col, inplace=True)                 # inplace=True permanently changes the dataframe


# - The total number of rows in the data are 1000 and there are three features which have a very large number of rows with missing values 
# - Those features are collision_type, property_damage, and policy_report_available. These are categorical attributes so instead of removing a very large number of rows from the data, we are just going to drop these three columns. 

# ### General Information on the Cleaned dataset:

# In[12]:


# this is the reduced number of features in the data
print(claims_clean.shape)


# In[13]:


# display general information about the data set
# good way to check if any features have null values and the levels of categorical variables
pd.DataFrame({'Dtype':claims_clean.dtypes,
              'Levels':[claims_clean[x].unique() for x in claims_clean.columns],
              'Null_Count':claims_clean.isna().sum(),
              'Number_Unique_Values':claims_clean.nunique()
             })


# The total number of rows and columns in the cleaned data 1000 x 30. We can see in the table that there are no missing values in the data and all the features are converted into appropriate types.

# ### Basic statistics of features on cleaned data:
# Let's explore the general characteristics of the data as a whole: examine the means, standard deviations, and other statistics associated with the numerical attributes.

# In[14]:


# summary of the distribution of features
claims_clean.describe(include="all").T


# The target feature (‘fraud_reported’) has a class imbalance (N=No Fraud Reported has 753 instances) whereas (Y=Yes Fraud Reported has 247 instances).
# Above is the table which displays the total count of instances, number of unique features,  top categorical value and frequency of that value, and some statistical measures such as Standard Deviation, Minimum value, 25% percentile, 50% percentile, 75% percentile, and the maximum value of each feature.

# ## Visualization of Data

# 
# ### Distribution of all categorical features in the data
# Show the distributions of values associated with categorical attributes using SEABORN package and/or plotting capabilities of Pandas to generate bar charts showing the distribution of categories for each attribute).

# In[15]:


# extract names of the categorical features in the data
categorical = claims_clean.select_dtypes(exclude=['float','int64']).columns
print(categorical)


# In[16]:


fig, ax = plt.subplots(5, 4, figsize=(20, 30))
fig.subplots_adjust(hspace=0.8, wspace=0.2)

for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(claims_clean[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# ### Cross Tabulated View using Barplots - Comparing categorical attributes with respect to target attribute (fraud_reported)
# Perform a cross-tabulation of all categorical attributes with the target attribute (fraud_reported). This requires the aggregation of the occurrences of each fraud_reported value (Y or N) separately for each value of the other attributes. Here, we are using SEABORN visualization package to create bar charts graphs to visualize of the relationships between these sets of variables.

# In[17]:


categorical2 = ['policy_state', 'policy_csl', 'policy_deductable', 'insured_sex',
                'insured_education_level', 'insured_relationship', 'incident_type', 
                'incident_severity', 'authorities_contacted', 'incident_state',
                'incident_city', 'number_of_vehicles_involved', 'bodily_injuries',
                'witnesses', 'auto_make', 'incident_part_of_the_day']


# In[18]:


fig, ax = plt.subplots(6, 3, figsize=(20, 35))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

for variable, subplot in zip(categorical2, ax.flatten()):
    sns.countplot(claims_clean['fraud_reported'], hue=variable, data=claims_clean, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# Following are some of the key points observed from these visualizations:
# - Policy_state: All of the claims or instances are from the three states (OH, IN, IL) and OH state has the highest number of fraud in claims compared to the other two states (IN, IL).
# - Insured_relationship: This is the relationship of the person with the policyholder who committed the fraud in insurance claims. It looks like that the category ‘other-relative’ has significantly higher fraud in claims compared to the other insured_relationships (husband, wife, own-child etc.) .
# - Incident_type: Single Vehicle Collisions and Multi Vehicle Collisions have significantly higher number of fraud in claims compared to the Vehicle Theft and Parked Car. This actually makes sense because in collision type claims, there is not only the damage to the vehicle but also the bodily injuries sustained to the driver and the passengers. The claims amount for the bodily injuries sustained is usually higher than just the property damage.
# - Incident_severity: We can observe that there is a huge spike in vehicle claims fraud when it comes to the severity of the incident. People tend to commit more fraud when there is a major damage to the vehicle. One of the reasons for this could be that the person does not want to commit a fraud when there is not too much damage or claim amount involved. 
# - Number_of_vehicles_involved: This is surprising to know that there is more fraud in claims when only one vehicle is involved in the accident. It may be easier to hide the facts and figures when it comes to just reporting single vehicle collisions.
# - Incident_part_of_the_day: Most of the accidents occurred during midnight or early morning when the fraud is detected in claims.

# ### Visualize Numerical Attributes
# 
# Let's visualize some of the numerical attributes using pairplot function from SEABORN package:
# 
# A pairplot plots a pairwise relationships in a dataset which shows the Correlation between numerical attributes and it also displays the distributions of all attributes. 

# In[19]:


numerical1 = ['months_as_customer', 'age', 'policy_annual_premium', 'capital-gains', 'capital-loss', 'auto_year']
numerical2 = ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']


# In[20]:


sns.pairplot(claims_clean[numerical1], corner=True)


# For this group of variables, we can see that the feature ‘age’ has a strong positive linear relationship with months_as_customer. There don't appear to be any significant outliers in this plot. However, if we look at the other features, there do not seem to be any positive or negative relationships among them so we are not too much concerned about the multicollinearity in the data.

# In[21]:


sns.pairplot(claims_clean[numerical2], corner=True)


# If we look at this group of features (total_claim_amount, injucy_claim, property_claim, vehicle_claim), they definitely have strong positive relationships among them. One of the reasons is that the total_claim_amount is the sum of injury_claim, property_claim, and vehicle_claim.

# ### Correlations Analysis
# Using the numeric attributes, lets perform basic correlation analysis among the attributes. The following Complete Correlation Matrix shows any significant positive or negative correlations among pairs of attributes. 
# 
# - A correlation matrix is a table showing correlation coefficients between sets of variables.
# - Correlation analysis is a very important step in pre-processing because it helps in identifying multicollinearity and building components in Principle Components Analysis (PCA).

# In[22]:


# Compute the correlation matrix
corr = claims_clean.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# The features that seem Strong Positively Correlated are (total_claim_amount, injucy_claim, property_claim, vehicle_claim). This is definitely a distinct group (orange squared boxes) of features.  This group is the same set of features which we identified above in the scatterplot matrix.

# ### Let's separate the target attribute ("fraud_reported") and the attributes used for model training¶
# The next step after cleaning and exploration of variables is to separate the target attribute ("fraud_reported") and the attributes used for model training. Let's create a separate data frame which contains the records without target attribute. Then pull the target attribute and store it separately.

# In[23]:


# records without target attribute
claims_records = claims_clean.drop('fraud_reported', 1)
claims_records.head()


# In[24]:


# Pull target attribute
# Let's separate the target attribute ("fraud_reported") from the attributes used for training
claims_target = claims_clean.fraud_reported
claims_target.head()


# ### Split the data into training and test sets (using 80%-20% randomized split). Note that the same split should also be performed on the target attribute).
# - Create a 20%-80% randomized split of the data. Set aside the 20% test portion; the 80% training data partition will be used for cross-validation on various tasks specified below.
# - We divide the data into randomized training and test partitions (note that the same split should also be perfromed on the target attribute). The easiest way to do this is to use the "train_test_split" module of "sklearn.cross_validation".

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(claims_records, claims_target, test_size=0.2, random_state=33)


# In[26]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ### Performing min-max normalization to rescale numeric features
# Let's use scikit learn preprocessing package for Min-Max Normalization to transform the values of all numeric attributes in the table onto the range 0.0-1.0. Fit the MinMaxScaler on the training data first and then transform the training and test data using this scaler.

# In[27]:


# names of the numerical features
nums = ['months_as_customer', 'age', 'policy_annual_premium', 
        'umbrella_limit', 'capital-gains', 'capital-loss', 
        'auto_year', 'total_claim_amount', 'injury_claim',
        'property_claim', 'vehicle_claim']


# In[28]:


# fit the MinMaxScaler on the training data
min_max_scaler = preprocessing.MinMaxScaler().fit(x_train[nums])

x_train[nums] = min_max_scaler.transform(x_train[nums])    # transform the training data using above scaler
x_test[nums] = min_max_scaler.transform(x_test[nums])      # transform the test data using above scaler


# In[29]:


# Let's look at the normalized training data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
x_train.head()


# In[30]:


# Let's look at the normalized test data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
x_test.head()


# 
# 
# 
# ### Convert the selected dataset into the Standard Spreadsheet format (i.e., convert categorical attributes into numeric by creating dummy variables)
# This requires converting each categorical attribute into multiple binary ("dummy") attributes (one for each values of the categorical attribute) and assigning binary values corresponding to the presence or not presence of the attribute value in the original record). The numeric attributes should remain unchanged.
# 
# 
# We first transformed the numerical features above because in general, it is best to avoid normalizing dummy variables. Even if the numerical values do not change, the underlying data type will be changed to float. Once converted to float, the variables are no longer treated as mutually exclusive binary values. It is not detrimental to KNN, but it can make the models lose the benefit of using one-hot encoding for many other ML algorithms (especially neural networks).

# In[31]:


# Applying pd.get_dummies to whole dataframe
# This will ignore int and float automatically and create dummies only for categorical attributes

x_train_ssf = pd.get_dummies(x_train) # standard spreadsheet format
x_test_ssf = pd.get_dummies(x_test) # standard spreadsheet format


# In[32]:


# Let's look at the converted training data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
x_train_ssf.head()


# In[33]:


# Let's look at the converted test data
np.set_printoptions(precision=2, linewidth=80, suppress=True)
x_test_ssf.head()


# ## Unsupervised Learning
# The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. In this section, we experimented with an unsupervised clustering technique (KMeans) using Scikit-learn implementation and also the dimensionality reduction technique called Principal Components Analysis (PCA).

# ## KMeans Clustering Usng Scikit-learn:
# 
# In this section, we are going to experiment with an unsupervised clustering technique (KMeans) using Scikit-learn implementation.
# 
# 
# K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. The objective of K-means clustering to group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.
# 
# 
# The K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

# ### Extract quantitative features from the data

# In[34]:


num_features = x_train_ssf[['months_as_customer', 'age', 'policy_annual_premium', 
                          'umbrella_limit', 'capital-gains', 'capital-loss', 
                          'auto_year', 'total_claim_amount', 'injury_claim',
                          'property_claim', 'vehicle_claim']]
num_features.head()


# In[35]:


# initialization
kmeans = KMeans(n_clusters=6,                       # no. of cluster
                max_iter=500,                       # max no. of iterations
                verbose=1)                          # gives different outputs


# In[36]:


kmeans.fit(num_features)                      # numerical dataframe

# it runs KMeans multiple times and each time it does with a different initialization
# at each iteration it goes through one iteration of the algorithm and it also outputs inertia
# the inertia is basically the sum of the squares of distances from the centroids of the clusters
# we want to have the smallest inertia and usually the best answer is at the bottom of this output


# ### Print the cluster centroids (use some formatting so that they are visually understandable).

# In[37]:


# display upto 2 decimal places
pd.options.display.float_format='{:,.2f}'.format

centroids = pd.DataFrame(kmeans.cluster_centers_,               # .cluster_centers_ function to get centroids
                         columns=num_features.columns)          # column names from the dataframe
centroids


# - First row is Mean vector associated with cluster 0
# - These are mean values for each one of the attributes
# - Centroids help us categorize different groups and see why certain attributes in different clusters

# In[38]:


# get the actual cluster assignments
# these are the cluster labels associated with each instance in the data
clusters = kmeans.predict(num_features)      # here we are not predicting anything but just to match the design pattern
                                                  # of other scikit learn function we write predict


# In[39]:


pd.DataFrame(clusters, columns=["Cluster"])
# each row in the data is assigned to one of the 7 clusters


# In[40]:


def cluster_sizes(data, clusters):
    #clusters is an array of cluster labels for each instance in the data
    
    size = {}
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]

    for c in cluster_labels:
        size[c] = len(data[clusters == c])
    return size


# In[41]:


size = cluster_sizes(num_features, clusters)

for c in size.keys():
    print("Size of Cluster", int(c), "= ", size[c])


# ### To evaluate clusters, first perform Silhouette analysis on the clusters (compute Silhouette values for all instances in the data, and then compute the overall mean Silhouette value.

# - One way to measure the quality of clustering is to compute the Silhouette values for each instance in the data.
# - The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
# - It is the ratio of the difference between in-cluster dissimilarity and the closest out-of-cluster dissimilarity, and the maximum of these two values.
# - The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and well separated from other clusters.
# - If most objects have a high value, then the clustering configuration is appropriate.
# - If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# In[42]:


# find silhouette values
silhouettes = metrics.silhouette_samples(num_features, clusters)       # pass the data and the clusters
print(silhouettes[:20])                                                # print first twenty values


# In[43]:


# mean of silhouettes values
# value closest to 1 is better
# we can use different number of k values and compare
print(silhouettes.mean())


# ### Visaulization of the Silhouettes

# In[44]:


def plot_silhouettes(data, clusters, metric='euclidean'):
    
    from matplotlib import cm
    from sklearn.metrics import silhouette_samples

    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = metrics.silhouette_samples(data, clusters, metric='euclidean')
    c_ax_lower, c_ax_upper = 0, 0
    cticks = []
    for i, k in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[clusters == k]
        c_silhouette_vals.sort()
        c_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        pl.barh(range(c_ax_lower, c_ax_upper), c_silhouette_vals, height=1.0, 
                      edgecolor='none', color=color)

        cticks.append((c_ax_lower + c_ax_upper) / 2)
        c_ax_lower += len(c_silhouette_vals)
    
    silhouette_avg = np.mean(silhouette_vals)
    pl.axvline(silhouette_avg, color="red", linestyle="--") 

    pl.yticks(cticks, cluster_labels)
    pl.ylabel('Cluster')
    pl.xlabel('Silhouette coefficient')

    pl.tight_layout()
    #pl.savefig('images/11_04.png', dpi=300)
    pl.show()
    
    return


# In[45]:


plot_silhouettes(num_features, clusters)


# - Some of the clusters are largest in size in terms of Silhouette coefficient
# - Dotted line is the mean silhouette value
# - Some clusters are signifcantly above the mean silhouette value

# ### Compute the Completeness and Homogeneity values of the generated clusters.

# In[46]:


print('Completeness : ', completeness_score(y_train, clusters))
print('Homogeneity  : ', homogeneity_score(y_train, clusters))


# - Homogeneity: each cluster contains only members of a single class.
# - Completeness: all members of a given class are assigned to the same cluster.
# 
# 
# - The completeness score approaches 1 when most of the data points that are members of a given class are elements of the same cluster.
# - While the homogeneity score approaches 1 when all the clusters contain almost only data points that are member of a single class.

# ## Principal Component Analysis (PCA) using Decomposition module in Scikit-learn
# Principal Component Analysis (PCA) is a statistical procedure that allows us to summarize the information contained in a large set by means of a smaller set of “summary indices” that can be more easily visualized and analyzed. It is a very common technique for “dimensionality reduction” and finding the “latent/hidden” factors from the data.
# 
# - The goal of PCA is to simplify model features into fewer, uncorrelated features to help visualize patterns in the data and help it run faster.
# - It reduces the number of variables while maintaining the majority of the important information.
# - It can help solve the very common problem of “Curse of Dimensionality”.
# - We should only apply PCA to continuous data and the data should be scaled before applying PCA technique.
# - PCA can be very helpful in predicting the parameter of interest.
# - PCA can help us build the parsimonious model which is easier to explain.
# - PCA can also be used in relation with other clustering techniques for better results.

# In[47]:


pca = decomposition.PCA(n_components=6)
claims_trans = pca.fit(num_features).transform(num_features)     # transforming the data


# In[48]:


np.set_printoptions(precision=2,suppress=True)
print(claims_trans)


# - The rows are still the claims data but the columns are the reduced # of features OR the principal components.
# - We can now use these components to cluster our claims data.
# - This is something similar to reduced reatures in classification.

# ### Analyze the principal components to determine the number of PCs needed to capture at least 90% of variance in the data.

# In[49]:


# what percentage of variance is explained by these five components
# eigenvalue / sum of all the eigenvalues
print(pca.explained_variance_ratio_)


# In[50]:


sum(pca.explained_variance_ratio_ * 100)


# - First PC explains 28% of variance in the data
# - Second PC explains 0.16% of variance in the data
# - They all combined represent about 90% of variance in the data
# - It can be obsereved that the first 6 components capture (explain) 90% of the variance in the data.

# ##### Provide a plot of PC variances.
# 
# - We can plot the principal components based on the percentage of variance they capture:

# In[51]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Cumulative Explained Variance')
plt.show()


# - These 6 components account for just over 90% of the variance. Using these 6 components, we can cover most of the essential characteristics of the data. 

# ### d): Kmeans on Transformed Data

# ##### Perform Kmeans again, but this time on the lower dimensional transformed data. 

# In[52]:


kmeans.fit(claims_trans)                        # transformed data


# In[53]:


# get the actual cluster assignments
# these are the cluster labels associated with each instance in the data
clusters_pca = kmeans.predict(claims_trans)    # here we are not predicting anything but just to match the design pattern
                                               # of other scikit learn function we write predict


# ##### Then compare Silhouette values as well as completeness and Homogeneity values of the new clusters. Compare these results with those obtained on the full data in part b.

# In[54]:


# find silhouette values
silhouettes_pca = metrics.silhouette_samples(claims_trans, clusters_pca)  # pass the data and the clusters
print(silhouettes_pca[:20])           # print first twenty values


# In[55]:


# mean of silhouettes values
# value closest to 1 is better
# we can use different number of k values and compare
print('Mean of silhouettes values on full data   : ', silhouettes.mean())
print('Mean of silhouettes values using PCA      : ', silhouettes_pca.mean())


# In[56]:


print('Completeness on full data                 : ', completeness_score(y_train, clusters))
print('Completeness using PCA                    : ', completeness_score(y_train, clusters_pca))
print()
print('Homogeneity on full data                  : ', homogeneity_score(y_train, clusters))
print('Homogeneity using PCA                     : ', homogeneity_score(y_train, clusters_pca))


# - We can observe from the above comparison that the Mean of silhouettes values improved using lower dimentional transformed data. It means that the quality of clustering has improved.
# - However, there is not a very big difference in the completeness and homogeneity values.

# The case for doing a cluster analysis (or other dimensionality reduction methods such as PCA) is probably to reduce the number of features in a way that the learning model is more robust. In case of this dataset, we do not have that many number of good numerical variables to group. 
# 
# 
# Clustering (unsupervised learning technique) comes into the play as our saviour when we do not have data to supervise. It is in general not a good idea to turn supervised learning (classification) into unsupervised learning (clustering) as it will sacrifice the vital information: labels.

# ## Predictive Modeling and Model Evaluation Using Classification Techniques 
# We are going to experiment with various classification models provided as part of the scikit-learn (sklearn) machine learning module using the pre-processed insurance claims data set. In particular, we will use the following list of classification algorithms to classify the data:
# 
# - K-Nearest-Neighbor (KNN) Classifier
# - Naive Bayes (Gaussian) classifier
# - Decision Tree Classifier
# - Linear Discriminant Analysis (LDA) Classifier
# - Rocchio Classifier
# - Random Forest Classifier
# - Ada Boost Classifier
# - Gradient Boosting Classifier
# - Support Vector Machines (SVC Classifier)
# - Logistic Regression Classifier
# 
# The basic objective is to create a best predictive model for classifying fraud, whether a claimant is likely to commit fraud in auto insurance claims. This is a pure classification task which also includes the misclassification costs.
# In addition to building the best model, we will also generate the confusion matrix (visualize it using Matplotlib), as well as the classification report.
# 
# 
# Using GridSearchCV from Scikit-learn, evaluate predictive models on 10-fold cross validation. We will also experiment with GridSearchCV for parameters optimization to see if we can improve accuracy (we will not provide the details of all of our experimentation, but will provide a short discussion on what parameters worked best as well as our final results).
# 
# 
# We are using Grid Search because it allows us to explore the parameter space more systematically and lets us select the best tuning parameters (aka "hyperparameters"). Grid Search allows us to define a grid of parameters that are searched using K-fold cross-validation.
# 
# 
# Finally, the accuracy of the different models will be compared to select the final model for prediction and the overview of model performances will be presented in a tabulated form.

# #### A versatile function to measure performance of a model

# In[57]:


def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)   
    
    if show_accuracy:
        Accuracy = metrics.accuracy_score(y, y_pred)
        #print("Accuracy: {0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("===============CLASSIFICATION REPORT===============")
        print(metrics.classification_report(y, y_pred),"\n")
      
    if show_confussion_matrix:
        print("===============CONFUSION MATRIX===============")
        cm = metrics.confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot = True, cmap = 'copper',fmt='g')
        plt.show()
        
    return Accuracy


# #### A versatile function to perform GridSearch on different models

# We are going to use the Grid Search to explore the parameter space more systematically.
# Select the best tuning parameters (aka "hyperparameters"). Grid Search allows us to define a grid of parameters that will be searched using K-fold cross-validation.

# In[58]:


# keeping track of the different statistics of all classification models
modeL = []
gsCV_BestScore = []
train_accuracy = []
test_accuracy = []


# In[59]:


def grid_search_function(m, modelName, params, xtrain, ytrain, xtest, ytest, cv=10):
    'Performs GridSearch and gets different statistics'
    
    gs = GridSearchCV(m, params, cv=10)             # perform Grid Search CV
    
    get_ipython().run_line_magic('time', '_ = gs.fit(xtrain, ytrain)                # print Wall Time of the model')
    print()
    print(gs.best_params_, gs.best_score_)          # print best parameters and best cv score
    print()
    
    train_acu = measure_performance(xtrain, ytrain, gs, show_accuracy=True, show_classification_report=False, show_confussion_matrix=False)
    test_acu = measure_performance(xtest, ytest, gs, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True)
    
    print("Training Accuracy: {0:.3f}".format(train_acu))          # print training accuracy
    print("Test Accuracy    : {0:.3f}".format(test_acu),"\n")      # print test accuracy
    
    modeL.append(modelName)                                        # add model name to the list
    gsCV_BestScore.append(gs.best_score_)                          # add best_score to the list 
    train_accuracy.append(train_acu)                               # add train accuracy to the list
    test_accuracy.append(test_acu)                                 # add test accuracy to the list


# ### KNN Classifier: 
# The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve the classification probelms. For each sample in data: It calculates the distance between the query example and the current example from the data. The algorithm is simple and easy to implement.

# In[60]:


knnclf = neighbors.KNeighborsClassifier()
print(knnclf.get_params())


# In[61]:


# define the parameter values that should be searched
parameters = {
    'weights': ['distance', 'uniform'],
    'n_neighbors': range(3, 11, 1),
}

# perform grid search using grid_search_function
grid_search_function(knnclf, 'K-Nearest-Neighbor', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Naive Bayes (Gaussian) Classifier:
# The Naïve Bayes Classifier belongs to the family of probability classifier, using Bayesian theorem. The reason why it is called ‘Naïve’ because it requires rigid independence assumption between input variables. Therefore, it is more proper to call Simple Bayes or Independence Bayes. This algorithm has been studied extensively since 1960s. Simple though it is, Naïve Bayes Classifier remains one of popular methods to solve text categorization problem, the problem of judging documents as belonging to one category or the other, such as email spam detection.
# 
# Bayes’s theorem plays a critical role in probabilistic learning and classification.
# - Uses prior probability of each class given no information about an item
# - Classification produces a posterior probability distribution over the possible classes given a description of an item
# - The models are incremental in the sense that each training example can incrementally increase or decrease the probability that a hypothesis is correct. Prior knowledge can be combined with observed data.

# In[62]:


nbclf = naive_bayes.GaussianNB()
print(nbclf.get_params())


# In[63]:


# define the parameter values that should be searched
parameters = {
    'priors': [None],
    'var_smoothing': [1e-09],
}

# perform grid search using grid_search_function
grid_search_function(nbclf, 'Naive Bayes (Gaussian)', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Decision Tree Classifier:
# Decision Tree algorithm belongs to the family of supervised learning algorithms. In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

# In[64]:


dt = tree.DecisionTreeClassifier()
print(dt.get_params())


# In[65]:


# define the parameter values that should be searched
parameters = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
    'max_depth': np.linspace(1, 10, 10, endpoint=True), 
    'min_samples_leaf': range(1,6),
}

# perform grid search using grid_search_function
grid_search_function(dt, 'Decision Tree', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# In[66]:


dtfinal = tree.DecisionTreeClassifier(criterion='gini', max_depth=3.0, min_samples_leaf=1, min_samples_split=0.30)
dtfinal.fit(x_train_ssf, y_train)

# call the predict function on the test intances to produce the predicted classes
treepreds = dtfinal.predict(x_test_ssf)

# print test accuracy
print('Test Accuracy: ', dtfinal.score(x_test_ssf, y_test))

# create a new dictionary with column names
dicT0 = {
    'gbpreds':treepreds, 
    'actual':y_test,
}

# create a dataframe using a dictionary
preds0 = pd.DataFrame(dicT0)
preds0.head()


# #### Visualizing the decision tree:

# In[67]:


fig, ax = plt.subplots(figsize=(50,20))
tree.plot_tree(dtfinal, feature_names=x_train_ssf.columns, class_names=["No","Yes"], filled=True, ax=ax);


# A decision tree is a flow-chart-like tree structure:
# - Internal node denotes a test on an attribute (feature)
# - Branch represents an outcome of the test
# - All records in a branch have the same value for the tested attribute
# - Leaf node represents class label or class label distribution

# In[68]:


import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(dtfinal,out_file=None, feature_names=x_train_ssf.columns, 
                           class_names=["No","Yes"], filled=True, rotate=True)
graph = graphviz.Source(dot_data)
graph


# #### Feature Importances

# In[69]:


(pd.Series(dtfinal.feature_importances_, index=x_train_ssf.columns)
   .nlargest(3)
   .plot(kind='barh')) 


# ### Linear Discriminant Analysis (LDA) Classifier:
# Linear discriminant analysis, normal discriminant analysis, or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics and other fields, to find a linear combination of features that characterizes or separates two or more classes of objects or events. Wikipedia

# In[70]:


ldclf = LinearDiscriminantAnalysis()
print(ldclf.get_params())


# In[71]:


# define the parameter values that should be searched
parameters = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': arange(0, 0.1, 0.001),    
}

# perform grid search using grid_search_function
grid_search_function(ldclf, 'Linear Discriminant Analysis', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Rocchio Classifier:
# The Rocchio algorithm is based on a method of relevance feedback found in information retrieval systems which stemmed from the SMART Information Retrieval System which was developed 1960-1964. Like many other retrieval systems, the Rocchio feedback approach was developed using the Vector Space Model. Wikipedia

# In[72]:


roclf = NearestCentroid()
print(roclf.get_params())


# In[73]:


# define the parameter values that should be searched
parameters = {
    'metric': ['euclidean', 'manhattan'],
    'shrink_threshold': arange(0, 1.01, 0.01),    
}

# perform grid search using grid_search_function
grid_search_function(roclf, 'Rocchio Classifier', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Random Forest Classifier (an example of bagging):
# - Each classifier in the ensemble is a decision tree classifier and is generated using a random selection of attributes at each node to determine the split.
# - During classification, each tree votes and the most popular class is returned.
# - Comparable in accuracy to Adaboost, but more robust to errors and outliers.
# - Insensitive to the number of attributes selected for consideration at each split, and faster than boosting.

# In[74]:


rf = RandomForestClassifier()
print(rf.get_params())


# In[75]:


# define the parameter values that should be searched
parameters = {
    'n_estimators': range(5, 101, 5),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
}

# perform grid search using grid_search_function
grid_search_function(rf, 'Random Forest', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Ada Boost Classifier:
# AdaBoost, short for “Adaptive Boosting”, is the first practical boosting algorithm proposed by Freund and Schapire in 1996. It focuses on classification problems and aims to convert a set of weak classifiers into a strong one.

# In[76]:


ab = AdaBoostClassifier()
print(ab.get_params())


# In[77]:


# define the parameter values that should be searched
parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0],
    'n_estimators': range(5, 51, 5),
}

# perform grid search using grid_search_function
grid_search_function(ab, 'Ada Boost', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Gradient Boosting Classifier:
# The following is an example of using Gradient Boosted Decision Trees. GBDT is a generalization of boosting to arbitrary differentiable loss functions. GBDT is an accurate and effective procedure that can be used for both regression and classification.
# 
# Analogy: Consult several doctors, based on a combination of weighted diagnoses—weight assigned based on the previous diagnosis accuracy
# 
# 
# How boosting works?
# - Weights are assigned to each training tuple
# - A series of k classifiers is iteratively learned
# - After a classifier Mi is learned, the weights are updated to allow the subsequent classifier, Mi+1 , to pay more attention to the training tuples that were misclassified by Mi 
# - The final M* combines the votes of each individual classifier, where the weight of each classifier's vote is a function of its accuracy
# 
# 
# Boosting algorithm can be extended for numeric prediction. Compared to bagging: Boosting tends to have greater accuracy, but it also risks overfitting the model to misclassified data

# In[78]:


gb = GradientBoostingClassifier()
print(gb.get_params())


# In[79]:


# define the parameter values that should be searched
parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0],
    'n_estimators': range(5, 51, 5),
    'random_state': [0],
}

# perform grid search using grid_search_function
grid_search_function(gb, 'Gradient Boosting', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# In[80]:


gbfinal = GradientBoostingClassifier(learning_rate=1.5, n_estimators=5, random_state=0)
gbfinal.fit(x_train_ssf, y_train)

# call the predict function on the test intances to produce the predicted classes
gbpreds = gbfinal.predict(x_test_ssf)

# print test accuracy
print('Test Accuracy: ', gbfinal.score(x_test_ssf, y_test))

# create a new dictionary with column names
dicT1 = {
    'gbpreds':gbpreds, 
    'actual':y_test,
}

# create a dataframe using a dictionary
preds1 = pd.DataFrame(dicT1)
preds1.head()


# #### Feature Importances

# In[81]:


(pd.Series(gbfinal.feature_importances_, index=x_train_ssf.columns)
   .nlargest(10)
   .plot(kind='barh')) 


# ### Support Vector Machines (SVC Classifier):
# Many classification models, such as Naïve Bayes, try to model the distribution of each class, and use these models to determine labels for new points. Sometimes called generative classification models.
# 
# SVMs  are discriminative classification models: Rather than modeling each class, they simply find a line or curve (in two dimensions) or a manifold (in multiple dimensions) that divides the classes from each other.

# In[82]:


svc = SVC()
print(svc.get_params())


# In[83]:


# define the parameter values that should be searched
parameters = {
    'degree': range(1,11),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 5, 10, 50, 100],
}

# perform grid search using grid_search_function
grid_search_function(svc, 'Support Vector Machines', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Logistic Regression Classifier:
# Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a 0 or 1.

# In[84]:


lr = LogisticRegression()
print(lr.get_params())


# In[85]:


# define the parameter values that should be searched
parameters = {
    'l1_ratio': np.linspace(0.0001, 100, 200),
    'penalty': ['l1', 'l2', 'elasticnet'],
}

# perform grid search using grid_search_function
grid_search_function(lr, 'Logistic Regression', parameters, x_train_ssf, y_train, x_test_ssf, y_test, cv=10)


# ### Performance of Different Models:

# In[86]:


# create a new dictionary with column names
dicT = {
    'Model':modeL, 
    'GSCV_BestScore':gsCV_BestScore, 
    'Train_Accuracy':train_accuracy,
    'Test_Accuracy':test_accuracy,
}

# create a dataframe using a dictionary
models_matrix = pd.DataFrame(dicT)

# set Model column as index of the dataframe
models_matrix = models_matrix.set_index('Model')

# Sort the dataframe by Test_Accuracy
models_matrix.sort_values(by=['Test_Accuracy'], ascending=False)


# ### Conclusion:

# In above table, the accuracy of the different models were compared to select the final model for prediction. The Gradient Boosting model was selected based on its performance. It had the highest test accuracy rate of 0.865. The future recommendation is to test and run this model in real time on a bigger dataset, since our dataset had only around 1000 instances. This model can flag potential fraud in insurance claims and can lead to reduction in insurance premiums, claim turnaround time, and increase company’s profitability. 

# In[ ]:





# In[ ]:




