package com.dev.bruno.ml.deeplearning.lstm

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SaveMode, SparkSession}

object ProcessCSV {

  def main(args: Array[String]): Unit = {

    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    // Basic configuration
    val conf = new SparkConf()
      .setAppName("ProcessCSVTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val spark = SparkSession.builder.config(conf).getOrCreate

    val dataSet = spark.read
      .option("header", "true") // The CSV file has header and use them as column names
      .option("inferSchema", "true").csv("./GOOG.csv")

    // Geting the NextOpenPrice for all dataSet
    dataSet.createOrReplaceTempView("temp_stocks")

    val nextOpenDataSetSql = "select date_add(Date, -1) as Date, Open as NextOpen, Close as NextClose, Low as NextLow, High as NextHigh, Volume as NextVolume from temp_stocks"

    val nextOpenDataSet = spark.sql(nextOpenDataSetSql)
    nextOpenDataSet.createOrReplaceTempView("temp_next_close_price")

    val sql = "select s.Date, s.Open, s.Close, s.Low, s.High, s.Volume, c.NextOpen, c.NextClose, c.NextLow, c.NextHigh, c.NextVolume from temp_stocks s, temp_next_close_price c where to_date(s.Date) = c.Date order by s.Date"
    val filter = "Date is not null and Open is not null and Close is not null and Low is not null and High is not null and Volume is not null and NextOpen is not null and NextClose is not null and NextLow is not null and NextHigh is not null and NextVolume is not null"

    val updatedDataSet = spark.sql(sql).filter(filter).distinct()

    // Saving the dataset as parquet
    updatedDataSet.write.mode(SaveMode.Overwrite).parquet("goog.parquet")

    spark.close()
  }
}
