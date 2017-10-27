
# coding: utf-8

# In[3]:


from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder     .master("local")     .appName("Predict Session Length for given IP")     .config("spark.executor.memory", "1gb")     .getOrCreate()
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
# sessionizing data based on 15 min fixed window time
# assign an Id to each session
SessionDF = mainDF.select(window("timestamp", "15 minutes").alias('FixedTimeWindow'),'timestamp',"ipaddress").groupBy('FixedTimeWindow','ipaddress').count().withColumnRenamed('count', 'NumberHitsInSessionForIp')
SessionDF = SessionDF.withColumn("SessionId", monotonically_increasing_id())
# join the time stamps and url to the Sessionized DF
dfWithTimeStamps = mainDF.select(window("timestamp", "15 minutes").alias('FixedTimeWindow'),'timestamp',"ipaddress","url")
SessionDF = dfWithTimeStamps.join(SessionDF,['FixedTimeWindow','ipaddress'])
# Finding the first hit time of each ip for each session and join in to our session df
FirstHitTimeStamps = SessionDF.groupBy("SessionId").agg(min("timestamp").alias('FristHitTime'))
SessionDF = FirstHitTimeStamps.join(SessionDF,['SessionId'])
timeDiff = (unix_timestamp(SessionDF.timestamp)-unix_timestamp(SessionDF.FristHitTime))
SessionDF = SessionDF.withColumn("timeDiffwithFirstHit", timeDiff)
tmpdf = SessionDF.groupBy("SessionId").agg(max("timeDiffwithFirstHit").alias("SessionDuration"))
SessionDF = SessionDF.join(tmpdf,['SessionId'])


# for any given IP if we don't have any previous log from that IP
# the prediction for it's session length is the average session length
meandf = SessionDF.groupBy().avg('SessionDuration')
meandf.show()


# In[4]:


# if the given ip has a record in the following table
# the prediction for it's session length is the it's previous session's average
meanSessionIP = SessionDF.groupBy("ipaddress").agg(avg("SessionDuration").alias('AverageSessionDurationForIP'))
meanSessionIP.show(20)


# In[6]:


# For Predicting the number of unique url visits for a given IP
# again if we don't have the IP in our logs, the predicted value is the
# average unique url visit by any IP
SessionDF.groupBy("ipaddress","url").count().distinct().groupBy().agg(avg("count")).show()


# In[8]:


# if we have the give ip in the records
# we can find the average previous unique url visits of that IP in the following table
SessionDF.groupBy("ipaddress","url").count().distinct().groupBy("ipaddress").agg(avg("count")).show()


# In[ ]:




