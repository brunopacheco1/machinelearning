package com.dev.bruno.ml.util

import java.io.File

import scala.collection.mutable.ListBuffer
import com.dev.bruno.ml.model.Ema
import com.dev.bruno.ml.model.Stock
import org.apache.spark.SparkConf
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SparkSession

object EmaCalculationTask {

  def calcEma(nameIndex: Double, stocks: Iterable[Stock]): List[Ema] = {
    // N = 180 days or 6 months
    val ema6Days = 180
    // N = 300 days or 10 months
    val ema10Days = 300
    val ema6Multiplier = 2D / (ema6Days + 1D)
    val ema10Multiplier = 2D / (ema10Days + 1D)
    val emas = new ListBuffer[Ema]()
    var counter = 1
    var previousEma6 = 0D
    var previousEma10 = 0D

    for (stock <- stocks) {
      var ema6 = 0D
      var ema10 = 0D

      if (counter <= ema6Days) previousEma6 += stock.close

      if (counter <= ema10) previousEma10 += stock.close

      if (counter == ema6Days) {
        previousEma6 /= ema6Days
        ema6 = previousEma6
      }
      else if (counter > ema6Days) {
        previousEma6 = (stock.close - previousEma6) * ema6Multiplier + previousEma6
        ema6 = previousEma6
      }

      if (counter == ema10) {
        previousEma10 /= ema10
        ema10 = previousEma10
      }
      else if (counter > ema10) {
        previousEma10 = (stock.close - previousEma10) * ema10Multiplier + previousEma10
        ema10 = previousEma10
      }

      emas += new Ema(nameIndex, ema6, ema10, stock.date)

      counter += 1
    }

    emas.toList
  }

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
      .setAppName("EmaCalculationTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val sqlContext = SparkSession.builder.config(conf).getOrCreate

    import sqlContext.implicits._

    val stocks = sqlContext.read.parquet("stocks.parquet").sort("Date").as(Encoders.bean(classOf[Stock])).rdd

    val stocksByCompany = stocks.groupBy(el => el.nameIndex)

    val emas = stocksByCompany.flatMap[Ema](el => calcEma(el._1, el._2))

    //val dataset = emas.toDF

    // Saving the dataset as parquet
    //dataset.write.mode(SaveMode.Overwrite).parquet("emas.parquet")

    sqlContext.close()
  }
}