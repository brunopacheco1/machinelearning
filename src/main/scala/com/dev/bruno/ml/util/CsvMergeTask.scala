package com.dev.bruno.ml.util

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql._

object CsvMergeTask {

  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("Please inform as args the location of CSV files.")
      return
    }

    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    // Basic configuration
    val conf = new SparkConf()
      .setAppName("CsvMergeTask")
      .setMaster("local[*]")

    // Initialization Spark SQL Context
    val sqlContext = SparkSession.builder.config(conf).getOrCreate

    val sparkContext = sqlContext.sparkContext

    val reader = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // The CSV file has header and use them as column names
      .option("inferSchema", "true") // Discover the column types

    //Loading CSV files directory
    val filter: String = "Open is not null and High is not null and Low is not null " + " and Volume is not null and Date is not null and Name is not null"
    val dataset = reader.load(args(0)).filter(filter).distinct()

    // Creating a index to use Name as a feature on Linear Regression
    val indexer = new StringIndexer()
      .setInputCol("Name")
      .setOutputCol("NameIndex")

    val indexedDataset = indexer.fit(dataset).transform(dataset)

    // Saving the dataset as parquet
    indexedDataset.write.mode(SaveMode.Overwrite).parquet("stocks.parquet")
  }
}