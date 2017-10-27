from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("Sessionize IP addresses") \
    .config("spark.executor.memory", "1gb") \
    .getOrCreate()

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