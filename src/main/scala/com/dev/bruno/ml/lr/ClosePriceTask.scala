package com.dev.bruno.ml.lr

import java.io.File
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object ClosePriceTask {


  def main(args: Array[String]): Unit = {
    val source = new File("./stocks.parquet")

    if (!source.exists || !source.isDirectory) {
      System.out.println("Please execute CsvMergeTask class before.")
      return
    }

    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    // Basic configuration
    val conf = new SparkConf()
      .setAppName("ClosePriceTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val sqlContext = SparkSession.builder.config(conf).getOrCreate

    // Columns to be use as input in Linear Regression Algorithm
    val features = Array("Open", "High", "Low", "NameIndex")

    // It is necessary to aggregate all features in one array
    // to use Linear Regression Algorithm
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val dataset = sqlContext.read.parquet("stocks.parquet")

    val featuredDataset = assembler.transform(dataset).sort("Date")

    // Split our dataset in two random ones for training and testing
    val trainingDataset = featuredDataset.filter("Date <= '2016-12-31'")
    val testDataset = featuredDataset.filter("Date > '2016-12-31'")

    // Linear Regression Algorithm
    // TODO Try to understand why we need to use setLabelCol
    val linearRegression = new LinearRegression()
      .setLabelCol("Close")
      .setFeaturesCol("features")
      .setPredictionCol("ClosePredicted")

    // Our training model to use in prediction
    val model = linearRegression.fit(trainingDataset)

    // A new column called prediction will be included in testDataset
    val predictedDataset = model.transform(testDataset)

    // Selecting only important columns to compare and show
    predictedDataset.select("Date", "Name", "Close", "ClosePredicted").show()

    sqlContext.close()
  }
}