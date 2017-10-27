
# coding: utf-8

# In[3]:


from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder     .master("local")     .appName("Machine learning for Load Prediction")     .config("spark.executor.memory", "1gb")     .getOrCreate()
sc = spark.sparkContext
# Load the data by creating rdd
rdd = sc.textFile('/home/hassan/Side_Projects/WeblogChallenge/data/2015_07_22_mktplace_shop_web_log_sample.log')
# split the data into columns
rdd = rdd.map(lambda line: line.split(" "))
# ====================================
# Manipulating data
# ====================================
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

#Map the RDD to a DF for better performance
mainDF = rdd.map(lambda line: Row(timestamp=line[0], ipaddress=line[2].split(':')[0],url=line[12])).toDF()
# convert timestamps from string to timestamp datatype
mainDF = mainDF.withColumn('timestamp', mainDF['timestamp'].cast(TimestampType()))

#get count of hit within window of 60 every seccond
loadperMinDF = mainDF.select(window("timestamp", "60 seconds").alias('timewindow'),'timestamp',"ipaddress").groupBy('timewindow').count().withColumnRenamed('count', 'HitperMin')
# get count of hit per IP
countdDF = mainDF.select(window("timestamp", "60 seconds").alias('timewindow'),'timestamp',"ipaddress").groupBy('timewindow','ipaddress').count().withColumnRenamed('count', 'HitperMin')
countdDF.show(20)


# In[4]:


# computing mean ,std and max of hit counts per IP within 60 seccond window as 
# features of each 60 seccond window 
# these features can be used for perdicting the next load in next minute
Feature1 = countdDF.groupBy("timewindow").agg(stddev('HitperMin').alias("stdOfHitPerMinPerIP"))
Feature2 = countdDF.groupBy("timewindow").agg(mean('HitperMin').alias("meanOfHitPerMinPerIP"))
Feature3 = countdDF.groupBy("timewindow").agg(max('HitperMin').alias("maxOfHitPerMinPerIP"))

Features = Feature1.join(Feature2,["timewindow"])
Features = Features.join(Feature3,["timewindow"])
Features = Features.join(loadperMinDF,["timewindow"])
Features = Features.orderBy('timewindow', ascending=True)
Features.show(20,False)


# In[5]:


# Divide hit coutns by 60 to become hit per seccond
Features = Features.withColumn("stdOfHitPerSecPerIP", col("stdOfHitPerMinPerIP")/60.0)    .withColumn("meanOfHitPerSecPerIP", col("meanOfHitPerMinPerIP")/60.0)    .withColumn("maxOfHitPerSecPerIP", col("maxOfHitPerMinPerIP")/60.0)    .withColumn("HitperSec", col("HitperMin")/60.0)
Features = Features.select("timewindow","stdOfHitPerSecPerIP","meanOfHitPerSecPerIP","maxOfHitPerSecPerIP","HitperSec")
Features.show(5)


# In[6]:


# get id for each window
Features = Features.withColumn("tagId", monotonically_increasing_id().cast("double"))
Features.show(10)


# In[7]:


# get hit per sec of each next 60 sec window 
# we will use this as the label for training model
from pyspark.sql.window import Window
w = Window.orderBy('tagId').rowsBetween(1,1) 
avgHit = avg(Features['HitperSec']).over(w) 
NextloadDf = Features.select(Features['stdOfHitPerSecPerIP'],Features['meanOfHitPerSecPerIP'],Features['maxOfHitPerSecPerIP'],Features['HitperSec'],avgHit.alias("LoadInNextOnedMin"))
NextloadDf.show(10)


# In[8]:


# removing null values
NextloadDf = NextloadDf.na.drop(subset=["stdOfHitPerSecPerIP"])
NextloadDf = NextloadDf.na.drop(subset=["LoadInNextOnedMin"])
NextloadDf = NextloadDf.na.drop(subset=["HitperSec"])

from pyspark.ml.linalg import DenseVector
# Define the `input_data` 
input_data = NextloadDf.rdd.map(lambda x: (x[4], DenseVector(x[:4])))
dataFrameInputdata = spark.createDataFrame(input_data, ["label", "features"])
dataFrameInputdata.first()


# In[22]:


# training a linear regression Model
train_data, test_data = dataFrameInputdata.randomSplit([.8,.2])
from pyspark.ml.regression import LinearRegression,RandomForestRegressor

rf = RandomForestRegressor(numTrees=100, maxDepth=10)
linearModel = rf.fit(train_data)

#lr = LinearRegression(labelCol="label", maxIter=100, regParam=0.3, elasticNetParam=0.8)
# Fit the data to the model
#linearModel = lr.fit(train_data)


predicted = linearModel.transform(test_data)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])
predictionAndLabel = predictions.zip(labels).collect()
  


# In[23]:


import numpy as np
error =[]
for a in predictionAndLabel:
   error.append(np.abs(a[0]-a[1]))
    
print 'mean abs error is: ',np.mean(error)


# In[28]:


# predicting the load for the next minute  
# becuase the last data record is the last 60 seccond of data
# we predict for the last record for predicting the next minute load

with_id = NextloadDf.withColumn("_id", monotonically_increasing_id())
i = with_id.select(max("_id")).first()[0]
last_item = with_id.where(col("_id") == i).drop("_id")
input_data = last_item.rdd.map(lambda x: (x[4], DenseVector(x[:4])))
dataFrameInputdata = spark.createDataFrame(input_data, ["label", "features"])
predicted = linearModel.transform(dataFrameInputdata)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])
predictions = predictions.collect()
predictions[:]


# In[ ]:




