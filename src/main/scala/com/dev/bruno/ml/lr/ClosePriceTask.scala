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
      .setAppName("LR.ClosePriceTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val spark = SparkSession.builder.config(conf).getOrCreate

    // Columns to be use as input in Linear Regression Algorithm
    val features = Array("NameIndex", "Open", "Close", "Low", "High", "Volume")

    // It is necessary to aggregate all features in one array
    // to use Linear Regression Algorithm
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val dataset = spark.read.parquet("stocks.parquet")

    // Geting the NextOpenPrice for all dataset
    dataset.createOrReplaceTempView("temp_stocks")

    val nextCloseDatasetSql = "select date_add(Date, -1) as Date, Close as NextClose from temp_stocks "

    val nextCloseDataset = spark.sql(nextCloseDatasetSql)
    nextCloseDataset.createOrReplaceTempView("temp_next_close_price")

    val sql = "select s.NameIndex, s.Name, s.Date, s.Open, s.Close, s.Low, s.High, s.Volume, c.NextClose from temp_stocks s, temp_next_close_price c where to_date(s.Date) = c.Date order by s.Date"
    val filter = "NameIndex is not null and Name is not null and Date is not null and Open is not null and Close is not null and Low is not null and High  is not null and Volume is not null and NextClose is not null"

    val updatedDataset = spark.sql(sql).filter(filter).distinct().sort("Date")

    val featuredDataset = assembler.transform(updatedDataset)

    // Split our dataset in two random ones for training and testing
    val trainingDataset = featuredDataset.filter("Date <= '2016-12-31'")
    val testDataset = featuredDataset.filter("Date > '2016-12-31'")

    // Linear Regression Algorithm
    val linearRegression = new LinearRegression()
      .setLabelCol("NextClose")
      .setFeaturesCol("features")
      .setPredictionCol("NextClosePredicted")

    // Our training model to use in prediction
    val model = linearRegression.fit(trainingDataset)

    // A new column will be included in testDataset
    val predictedDataset = model.transform(testDataset)

    // Selecting only important columns to compare and show
    predictedDataset.select("Date", "Name", "NextClose", "NextClosePredicted").show()

    spark.close()
  }
}