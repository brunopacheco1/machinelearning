package com.dev.bruno.ml.clustering

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object Kmeans {

  def main(args: Array[String]): Unit = {
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    val conf = new SparkConf()
      .setAppName("Kmeans")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val dataSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/mall_customers.csv")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Annual Income (k$)", "Spending Score (1-100)"))
      .setOutputCol("features")

    val clustering = new KMeans()
      .setFeaturesCol("features")
      .setK(5)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, clustering))

    val model = pipeline.fit(dataSet)

    val dataSetTransformed = model.transform(dataSet)

    dataSetTransformed.show()

    spark.close()
  }
}