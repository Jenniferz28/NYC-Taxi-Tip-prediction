# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:38:28 2016

@author: qz
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

'''1: set working directory and load data'''

data = pd.read_csv("green_tripdata.csv",sep=",",header=0)
data.shape # there are 21 columns and ~1.5 milliion rows

data.head()   #check each fearture
pd.DataFrame(data.columns) #list column names
data.isnull().sum() #check the number of missing values

'''2. Plot a histogram of Trip Distance
Apparently the most of trip_dis cumulates below 100. But it is not quite straightforward to see the pattern.
Let’s show kernel density estimate:The histogram and frequency plot tells us that the trip distance is
rightskewed distribution,Mean distance is larger than median distance. This indicates that appropriate 
transformation of this variable (for example, BoxCox transformation) is needed if we are using linear model'''

data['Trip_distance'].describe()
trip_dis=data['Trip_distance']
plt.figure(1)
plt.hist(trip_dis, alpha=.3)
plt.title("Trip Distance Histgram")
plt.xlabel("Trip Distance")
plt.ylabel("Numbers")

# let's show kernel density estimate
seaborn.distplot(trip_dis) # most of the data is with in [0,50] range
seaborn.distplot(trip_dis[trip_dis <=30.0])
#Let's take a closer look at the distribution below 30.0
plt.figure(2)
plt.hist(trip_dis[trip_dis <=30.0],bins=20)
plt.title("Trip Distance Histgram")
plt.xlabel("Distance")
plt.ylabel("numbers") # The histogram structure is right-skewed distribution. Box-cox transformation is needed.


'''3.Let's look at the Mean and Median trip distance grouped by hour of ay
Get a sense of identifying trips that coming from or arriving at the NYC area airports '''

# convert object to timestamp
T_pick=pd.to_datetime(data['lpep_pickup_datetime'],box=True)
T_drop=pd.to_datetime(data['Lpep_dropoff_datetime'],box=True)
T_delta=T_drop-T_pick
# Split out the Hour .
hour_pick=pd.DatetimeIndex(T_pick).hour #hour of pick up
hour_drop=pd.DatetimeIndex(T_drop).hour #hour of drop off
# Test if these two are the same
hour_pick ==hour_drop #it's hard to tell if every single element is the samle
(hour_pick ==hour_drop).all() #False.

#Add the hour_drop and hour-pick into the data frame'''
data['hour_pick']=hour_pick
data['hour_drop']=hour_drop
#mean and median trip distance grouped by hour-pick'''
mean_pick=data.groupby('hour_pick', as_index=False)['Trip_distance'].mean()
median_pick=data.groupby('hour_pick', as_index=False)['Trip_distance'].median()
# mean and median trip distance grouped by hour-drop'''
mean_drop=data.groupby('hour_drop', as_index=False)['Trip_distance'].mean()
median_drop=data.groupby('hour_drop', as_index=False)['Trip_distance'].median()

(mean_pick[[1]] == mean_drop[[1]]).all() #False
(median_pick[[1]] == median_drop[[1]]).all() #False

#Boxplot to compare Mean and Median in pickup hour
m1=[mean_pick['Trip_distance'],median_pick['Trip_distance']]
plt.boxplot(m)
plt.xticks([1, 2],['Mean', 'Median'])
plt.ylabel('(Pickup) Trip Distance')
#Plot the mean and median
plt.figure(3)
plt.plot(mean_pick['hour_pick'],mean_pick['Trip_distance'],'bs--', label='Mean trip distance')
plt.plot(median_pick['hour_pick'],median_pick['Trip_distance'], 'r^--', label='Median trip distance')
plt.title("Pickup Trip Distance vs. Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Trip Distance")
plt.legend(loc='best',fontsize=16)

#The hourly mean_pick and mean_drop, hourly median_pick and median_drop are very different.
#Boxplot to compare Mean and Median in dropoff hour
m2=[mean_drop['Trip_distance'],median_drop['Trip_distance']]
plt.boxplot(m)
plt.xticks([1, 2],['Mean', 'Median'])
plt.ylabel('(Drop-off) Trip Distance')
#Plot the mean and median for drop-off hour
plt.figure(4)
plt.plot(mean_drop['hour_drop'],mean_drop['Trip_distance'],'bs--', label='Mean trip distance')
plt.plot(median_drop['hour_drop'],median_drop['Trip_distance'], 'r^--', label='Median trip distance')
plt.title("Drop-off Trip Distance vs. Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("(Drop-off)Trip Distance")
plt.legend(loc='best',fontsize=16)

'''Here is the information we get from the plots:
#1.Generally the relatively long-distance-trip of green taxi takes place during 9pm – 0am, 4am - 7 am. People
at NYC take green taxi for longdistance trip at 5pm -6pm
#2.The relatively short-distance trip time range is 10 am-7pm, which makes sense because that is the usual work
and daily operation time,probably it would be associated with local traffic.Hourly mean is always larger than
hourly median. Most of the people taking green taxi prefer short-distance trip (<2.5)'''

'''
NYC area Airport Coordinates Definition
* JFK: 40.6413° N, 73.7781° W ;(-73.7781,40.6413)
* LaGuardia: 40.7769° N, 73.8740° W; (-73.8740,40.7769)
* Newark Liberty: 40.6895° N, 74.1745° ;(-74.1745,40.6895)
*Republic;40.7261° N, 73.4168° W ; (-73.4168,40.7261)
'''
# Here for the sake of simplicity, we assume the area +/- 0.0001 portion of lattitude and longitude is near the airpot
data['Pickup_longitude'].describe()
data['Pickup_latitude'].describe()
data['Dropoff_longitude'].describe()
data['Dropoff_latitude'].describe()
Airport=np.column_stack(([-73.7781,-73.8740,-74.1745],
                        [40.6413,40.7769,40.6895]))
J=[-73.7781,40.6413]
L=[-73.8740,40.7769]
N=[-74.1745,40.6895]
R=[-73.4168,40.7261]


# JFK
JFK=data.query('(1.0001*(-73.7781)<Pickup_longitude < 0.9999*(-73.7781) and   0.9999*40.6413<Pickup_latitude<1.0001*40.6413) or (1.0091*(-73.7781)<Dropoff_longitude < 0.9999*(-73.7781) and 0.9999*40.6413<Dropoff_latitude<1.0001*40.6413)')
JFK.shape[0] # 18201

#LGA
LGA=data.query('(1.0001*(-73.8740)<Pickup_longitude < 0.9999*(-73.8740) and   0.9999*40.7769<Pickup_latitude<1.0001*40.7769) or (1.0001*(-73.8740)<Dropoff_longitude < 0.9999*(-73.8740) and 0.9999*40.7769<Dropoff_latitude<1.0001*40.7769)')
LGA.shape[0] #11265

#NL
NL=data.query('(1.0001*(-74.1745)<Pickup_longitude < 0.9999*(-74.1745) and   0.9999*40.6895<Pickup_latitude<1.0001*40.6895) or (1.0001*(-74.1745)<Dropoff_longitude < 0.9999*(-74.1745) and 0.9999*40.6895<Dropoff_latitude<1.0001*40.6895)')
NL.shape[0] # 268

#Republic
Rep=data.query('(1.0001*(-73.4168)<Pickup_longitude < 0.9999*(-73.4168) and   0.9999*40.7261<Pickup_latitude<1.0001*40.7261) or (1.0001*(-73.4168)<Dropoff_longitude < 0.9999*(-73.4168) and 0.9999*40.7261<Dropoff_latitude<1.0001*40.7261)')
Rep.shape[0] # 1

#let's Look at JFK characteristics
JFK['Fare_amount'].describe() # Average fare is 33.25
JFK['Fare_amount'].plot.hist(alpha=0.5) # There is negative values
plt.title('Fare_amount JFK:average=33.25')

JFK[JFK['Fare_amount']<0]   # 10  observations have negative

#JFK[['Fare_amount']][JFK['Fare_amount']> 0].describe()
JFK['Payment_type'].plot.hist(alpha=0.5)
plt.title('Payment_type') #Payment_type


JFK.boxplot(column='Tip_amount',by='Payment_type') # The tip_amount depends on the Payment_type
JFK.boxplot(column='Fare_amount',by='Payment_type') # The fare_amount doesn't dependent on Payment_type.
JFK.boxplot(column='Tip_amount',by='RateCodeID') # The tip_amount seems not related to RateCodeID
JFK.boxplot(column='Tip_amount',by='hourdrop')


#plot fare_amount vs trip_distance
plt.plot(JFK['Trip_distance'],JFK['Fare_amount'],'g.')
plt.title('JFK Fare_amount vs Trip_distance')
plt.xlabel('Trip_distance')
plt.ylabel('Fare_amount')

#plot Tip_amount vs trip_distance
plt.plot(JFK['Trip_distance'],JFK['Tip_amount'],'b.')
plt.title('JFK Tip_amount vs Trip_distance')
plt.xlabel('Trip_distance')
plt.ylabel('Tip_amount')

''' According to the 4 boxplots above, apparently at JFK area, the payment type has huge impact on the 
Tip_amount while the Fare_amount is relatively stable on payment type. The RateCodeID also makes the 
Tip_amount selective. Furthermore, the Tip_amount is dependent on the hour_drop ( or hour_pick) . 
The next is to look at the possible relationship between Tip_amount and travel distance. Overall, 
the Fare_amount and Tip_amount are positively increasing as the Trip_distrance increases. Which agree 
with common sense that people would spend more and tip more if they take longer-distance trip'''


#Look at the percentage of tip of transactions'''
tip=100*data['Tip_amount']/data['Total_amount']
tip.describe()
tip.isnull().sum() # there are 4172 NA values because Ttao_amout is 0
tip.plot.hist(alpha=0.7,color='r') #histgram of tip%
plt.title('Tip percentage histgram')
plt.xlabel('Tip%')
plt.ylabel('Frequency')

data['Tip_p']=tip

#little resaerch on Tip_amount
plt.hist(data['Payment_type'], alpha=.7)
plt.title('Payment_type histgram')

data.boxplot(column='Tip_p',by='Payment_type')

#plot of credit card tip%
data[data['Payment_type']==1].shape #(701287, 24)
data[data['Payment_type']==1].Tip_p.plot.hist(alpha=0.7,color='b',bins=10)
plt.title('Payment_type=1 Tip%')

#plot of cash payment tip%
data[data['Payment_type']==2].Tip_p.describe()# mean=0, n=780112
data[data['Payment_type']==2].Tip_p.plot.hist(alpha=0.7,color='m',bins=10)
plt.title('Payment_type=2 Tip%')

#other payment type
data[data['Payment_type']!=1&2].Tip_p.describe() # it seems like most of Tip_amount for non-credit is 0.

'''Most of the transactions are paid by credit card or cash. The transactions Payment_type=1 has broad 
distribution, while Payment_type=2 looks like only at one value. Let’s verify our hypothesis by plotting 
the individual histogram. Tip by cash payment is mostly 0, which makes sense because it was not recorded 
by the meter. For other payment types,it seems like the Tip_p is very likely to be 0 '''


'''4. A little research on the distribution of speed'''
data['t']=pd.DatetimeIndex(T_delta).hour + (pd.DatetimeIndex(T_delta).minute)/60.0 + (pd.DatetimeIndex(T_delta).second)/3600.0  #unit: Hours
data['t'].describe() # some of t is 0, may cause singularity
data['week']=pd.DatetimeIndex(T_pick).weekofyear
df5=data[data['t']>0]

df5['speed']= df5['Trip_distance'].div(df5.t,axis=0)

df5['speed'].describe() # the maximum is 2.026800e+05, which doens't make senses--outlier detection

# Let's look at the speed below 200.
df6=df5[df5['speed'] < 100.0]

#plot grouped box-plot
df6.boxplot(column='speed',by='week')

# calculate the mean
weekly_ave_speed=df6.groupby(['week'], as_index=False)['speed'].mean()
print weekly_ave_speed

'''The weekly average speed are the same. A multi-sample ANOVA could be used to test if the weekly average 
speed are equal. Test for normal distribution is needed before the ANOVA'''


#plot grouped box-plot.we show the speed grouped by hour in box-plot. The pattern of mean
#indicates probably the average hourly speed are not equal to each other.
df6.boxplot(column='speed',by='hour_pick')

hour_ave_speed=df6.groupby(['hour_pick'], as_index=False)['speed'].mean()
print hour_ave_speed


'''5. Model Building'''

#Before building the model:clean data, take care missing values,feature scaling and one-hot encode the categorical variables
# remove missing values and nan
df=data.drop(['lpep_pickup_datetime','Lpep_dropoff_datetime','Ehail_fee'],axis=1)
df.describe() # Trip_type and Tip_p have NaN

df0=df[df['Trip_type '].notnull() & df['Tip_p'].notnull()]#remove NaN observations


#change data the numbers to categorical
factor=["VendorID","RateCodeID","Payment_type","Trip_type ",'Store_and_fwd_flag']
for variable in factor:
    dummies = pd.get_dummies(df0[factor], prefix = variable)
    df1= pd.concat([df0, dummies],axis=1)
    df1.drop([variable], axis=1, inplace=True)

Tip_p= df1['Tip_p']          #only y
df2=df1.drop(['Tip_p'],axis=1) #all the x
X_train,X_test,Y_train,Y_test = train_test_split(df2,Tip_p,test_size=0.20)

gbr=GBR(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1, max_depth =6, verbose =1)
est=gbr.fit(X_train,Y_train)
#prediction experiment
y_test_pred=est.predict(X_test)
mean_squared_error(y_test_pred,Y_test)
#plot the train loss vs. iteration
n = np.arange(100)+1
plt.plot(n, est.train_score_,'r-')
plt.ylabel('Training Loss')
plt.xlabel('Iteration')

'''lowest mean square error 0.024.I tuned number of iterations ‘n_estimators’ :10 to200, ‘subsample’: 0.1 to 1,
‘max_depth’: 3 to 8. I tuned number of iterations ‘n_estimators’ :10 to200, ‘subsample’: 0.1 to 1, ‘max_depth’: 3 to 8.
Cross-validation and greed search could be used to tune the parameters in the GradientBoostingRegressor model.
This could be able to get better performance and higher accuracy.'''
