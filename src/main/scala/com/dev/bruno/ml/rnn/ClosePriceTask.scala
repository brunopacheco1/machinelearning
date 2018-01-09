package com.dev.bruno.ml.rnn

import java.io.File
import java.sql.Timestamp

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.SparkConf
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

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
      .setAppName("RNN.ClosePriceTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val spark = SparkSession.builder.config(conf).getOrCreate

    val sc = spark.sparkContext

    val dataset = spark.read.parquet("stocks.parquet")

    // Split our dataset in two random ones for training and testing
    val trainingDataset = dataset.filter("name = 'GOOG' and Date <= '2016-12-31'")

    val testDataset = dataset.filter("name = 'GOOG' and Date > '2016-12-31'")

    println("counters: training -> " + trainingDataset.count() + ", testing -> " + testDataset.count())

    val trainingData: RDD[DataSet] = trainingDataset.rdd.map[DataSet](price => createDataSet(price))

    val testingData: RDD[DataSet] = testDataset.rdd.map(price => createDataSet(price))

    val sparkNetwork = RNNBuilder.build(5, 1, 1, sc)

    val epochs = 100

    for(i <- 1 until epochs) {
      val net = sparkNetwork.fit(trainingData)
      net.rnnClearPreviousState()
    }

    val stats = sparkNetwork.getSparkTrainingStats()    //Get the collect stats information
    StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc)

    val net = sparkNetwork.getNetwork

    val locationToSave = new File("./StockPriceLSTM.zip")
    // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)

    testingData.foreach(dataset => {
      val result = net.rnnTimeStep(dataset.getFeatures).getDouble(0)

      println(result + ", " + dataset.getFeatures.getDouble(0))
    })
  }

  def createDataSet(price: Row): DataSet = {
    val Open = price.getDouble(1)
    val Close = price.getDouble(2)
    val Low = price.getDouble(3)
    val High = price.getDouble(4)
    val Volume = price.getDouble(5)

    val input = Nd4j.create(Array[Int](1, 5, 1), 'f')

    input.putScalar(Array[Int](0, 0, 0), Open)
    input.putScalar(Array[Int](0, 1, 0), Close)
    input.putScalar(Array[Int](0, 2, 0), Low)
    input.putScalar(Array[Int](0, 3, 0), High)
    input.putScalar(Array[Int](0, 4, 0), Volume)

    val label = Nd4j.create(Array[Int](1, 1, 1), 'f')

    label.putScalar(Array[Int](0, 0, 0), Close)

    new DataSet(input, label)
  }
}
