package com.dev.bruno.ml.lr

import java.io.File
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object NextOpenPriceTask {


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
      .setAppName("NextOpenPriceTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val sqlContext = SparkSession.builder.config(conf).getOrCreate

    val dataset = sqlContext.read.parquet("stocks.parquet")

    // Geting the NextOpenPrice for all dataset
    dataset.createOrReplaceTempView("temp_stocks")

    val nextOpenDatasetSql = "select date_add(Date, -1) as Date, " + "NameIndex, Open as NextOpenPrice from temp_stocks "

    val nextOpenDataset = sqlContext.sql(nextOpenDatasetSql)
    nextOpenDataset.createOrReplaceTempView("temp_next_openprice")

    val sql = "select s.*, o.NextOpenPrice from temp_stocks s, temp_next_openprice o" + " where to_date(s.Date) = o.Date and s.NameIndex = o.NameIndex"
    val updatedDataset = sqlContext.sql(sql)

    // Columns to be use as input in Linear Regression Algorithm
    val features = Array("Open", "Close", "High", "Low", "NameIndex")

    // It is necessary to aggregate all features in one array
    // to use Linear Regression Algorithm
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    // Linear Regration Algorithm
    // TODO Try to understand why we need to use setLabelCol
    val linearRegression = new LinearRegression
    linearRegression.setLabelCol("NextOpenPrice")
    linearRegression.setFeaturesCol("features")
    linearRegression.setPredictionCol("NextOpenPricePredicted")

    val featuredDataset = assembler.transform(updatedDataset).sort("Date")

    // Split our dataset in two random ones for training and testing
    val trainingDataset = featuredDataset.filter("Date <= '2016-12-31'")
    val testDataset = featuredDataset.filter("Date > '2016-12-31'")

    // Our training model to use in prediction
    val model = linearRegression.fit(trainingDataset)

    // A new column called prediction will be included in testDataset
    val predictedDataset = model.transform(testDataset)

    // Selecting only important columns to compare and show
    predictedDataset.select("Date", "Name", "NextOpenPrice", "NextOpenPricePredicted").show()

    sqlContext.close()
  }
}