// Databricks notebook source
import sys.process._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
"wget -P /tmp https://www.datacrucis.com/static/www/datasets/stratahadoop-BCN-2014.json" !!

// COMMAND ----------

val localpath="file:/tmp/stratahadoop-BCN-2014.json"
dbutils.fs.mkdirs("dbfs:/datasets/")
dbutils.fs.cp(localpath, "dbfs:/datasets/")
display(dbutils.fs.ls("dbfs:/datasets/stratahadoop-BCN-2014.json"))
val df = sqlContext.read.json("dbfs:/datasets/stratahadoop-BCN-2014.json")
val rdd = df.select("text").rdd.map(row => row.getString(0))

// COMMAND ----------

val wordCounts = rdd.flatMap(_.split(" ")).map(word => (word,1)).reduceByKey((a,b) => a+b)
wordCounts.take(10).foreach(println)

// COMMAND ----------

display(df)

// COMMAND ----------

val hashtags = df.select("entities")
//hashtags.take(1).foreach(x => println(x.hashtags)
df.select("entities").rdd.map(row => row.getString(0)).take(5)
df.limit(1).select("entities").as[String].collect()(0)
df.limit(1).select("entities").rdd.zipWithIndex.filter(_._2==1).map(_._1).first()

// COMMAND ----------

df.printSchema()

// COMMAND ----------

val hashtags = df.select("entities.hashtags.text").as[String].collect()
hashtags(0)

// COMMAND ----------

val hashtags = df.select("entities.hashtags.text").as[String]
val rdd2 = hashtags.rdd.map(word => word.slice(1, word.length - 1 ).toLowerCase().replaceAll("\\s", ""))
val wc2 = rdd2.
  flatMap(_.split(",")).
  map(word => (word,1)).
  reduceByKey((a,b) => a+b)

// COMMAND ----------

wc2.sortBy(-_._2).take(100).foreach(println)

// COMMAND ----------

wc2.sortBy(-_._2).take(10).foreach(println)

// COMMAND ----------

val users = df.select("user.id").as[String]
val rdd3 = users.rdd.map(row => row)
val wc3 = rdd3.map(word => (word,1)).reduceByKey((a,b) => a+b)
wc3.sortBy(-_._2).take(10).foreach(println)

// COMMAND ----------

val time = df.select("created_at")
users.take(10).foreach( e =>  println(e.getClass))

// COMMAND ----------

val all = df.select("created_at","entities.hashtags.text").as[(String,String)]
all.take(10).foreach(println)
val rdd4 = all.rdd.map(word => (word._1.slice(4,11) ,word._2.slice(1, word._2.length - 1 ).toLowerCase().replaceAll("\\s", "")))
val wc4 = rdd4.
  map(word => ((word._1,word._2),1)).
  reduceByKey((a,b) => a+b)

// COMMAND ----------

wc4.sortBy(r => (-r._2,r._1)).take(10).foreach(println)

// COMMAND ----------
