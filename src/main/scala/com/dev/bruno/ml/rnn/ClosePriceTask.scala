package com.dev.bruno.ml.rnn

import java.io.File

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.SparkConf
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object ClosePriceTask {

  def main(args: Array[String]): Unit = {
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

    val dataset = spark.read.format("com.databricks.spark.csv")
      .option("header", "true") // The CSV file has header and use them as column names
      .option("inferSchema", "true").load("./GOOG.csv")

    // Geting the NextOpenPrice for all dataset
    dataset.createOrReplaceTempView("temp_stocks")

    val nextOpenDatasetSql = "select date_add(Date, -1) as Date, Close as NextClose from temp_stocks "

    val nextOpenDataset = spark.sql(nextOpenDatasetSql)
    nextOpenDataset.createOrReplaceTempView("temp_next_close_price")

    val sql = "select s.Date, s.Open, s.Close, s.Low, s.High, s.Volume, c.NextClose from temp_stocks s, temp_next_close_price c where to_date(s.Date) = c.Date order by s.Date"
    val filter = "Date is not null and Open is not null and Close is not null and Low is not null and High  is not null and Volume is not null and NextClose is not null"

    val updatedDataset = spark.sql(sql).filter(filter).distinct().sort("Date")

    val min = updatedDataset.groupBy().min("Open", "Close", "Low", "High", "Volume").head()

    val max = updatedDataset.groupBy().max("Open", "Close", "Low", "High", "Volume").head()

    // Split our dataset in two random ones for training and testing
    val trainingDataset = updatedDataset.filter("Date <= '2017-12-22'")
    val testDataset = updatedDataset.filter("Date > '2017-12-22'")

    println("counters: training -> " + trainingDataset.count() + ", testing -> " + testDataset.count())

    val trainingData: RDD[DataSet] = trainingDataset.rdd.map[DataSet](price => createDataSet(price, min, max))

    val testingData: RDD[DataSet] = testDataset.rdd.map(price => createDataSet(price, min, max))

    val sparkNetwork = RNNBuilder.build(5, 1, 1, sc)

    val epochs = 100

    for(i <- 1 until epochs) {
      val net = sparkNetwork.fit(trainingData)
      net.rnnClearPreviousState()
    }

    //sparkNetwork.fit(trainingData)

    val stats = sparkNetwork.getSparkTrainingStats()    //Get the collect stats information
    StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc)

    val net = sparkNetwork.getNetwork

    val locationToSave = new File("./lstm_model.zip")
    // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)

    testingData.collect().foreach(dataset => {
      val prediction = net.rnnTimeStep(dataset.getFeatures)

      val result = prediction.getDouble(0) * (max.getDouble(1) - min.getDouble(1)) + min.getDouble(1)

      val actual = dataset.getLabels.getDouble(0) * (max.getDouble(1) - min.getDouble(1)) + min.getDouble(1)

      /*
      println("______________________PREDICTION______________________")
      println(prediction)
      println("______________________LABELS______________________")
      println(dataset.getLabels)
      println("______________________FEATURES______________________")
      println(dataset.getFeatures)
      println("______________________END______________________")
      */

      println(result + ", " + actual)
    })

    spark.close()
  }

  def createDataSet(price: Row, min: Row, max : Row): DataSet = {
    val Open = price.getDouble(1)
    val Close = price.getDouble(2)
    val Low = price.getDouble(3)
    val High = price.getDouble(4)
    val Volume = price.getInt(5).toDouble
    val NextClose = price.getDouble(6)

    val input = Nd4j.create(Array[Int](1, 5, 1), 'f')

    input.putScalar(Array[Int](0, 0, 0), (Open - min.getDouble(0)) / (max.getDouble(0) - min.getDouble(0)))
    input.putScalar(Array[Int](0, 1, 0), (Close - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))
    input.putScalar(Array[Int](0, 2, 0), (Low - min.getDouble(2)) / (max.getDouble(2) - min.getDouble(2)))
    input.putScalar(Array[Int](0, 3, 0), (High - min.getDouble(3)) / (max.getDouble(3) - min.getDouble(3)))
    input.putScalar(Array[Int](0, 4, 0), (Volume - min.getInt(4).toDouble) / (max.getInt(4).toDouble - min.getInt(4).toDouble))

    val label = Nd4j.create(Array[Int](1, 1, 1), 'f')

    label.putScalar(Array[Int](0, 0, 0), (NextClose - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))

    new DataSet(input, label)
  }
}
