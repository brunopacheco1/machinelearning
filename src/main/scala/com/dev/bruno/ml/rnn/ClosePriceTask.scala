package com.dev.bruno.ml.rnn

import java.io.File

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

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
    val sqlContext = SparkSession.builder.config(conf).getOrCreate

    val batchSize = 64 // mini-batch size

    val epochs = 100 // training epochs

    val sc = sqlContext.sparkContext

    val recordReader = new CSVRecordReader(1, ",")
    recordReader.initialize(new FileSplit(new File("./GOOG.csv")))

    //reader,label index,number of possible labels
    val iterator = new RecordReaderDataSetIterator(recordReader, batchSize)

    val trainDatasetList = ListBuffer[DataSet]()

    var lastDataSet: DataSet = null

    while(iterator.hasNext) {
      val item = iterator.next()
      lastDataSet = item

      trainDatasetList += item
    }

    println(trainDatasetList.size)

    val trainingData: RDD[DataSet] = sc.parallelize(trainDatasetList)

    val sparkNetwork = RNNBuilder.build(6, 1, sc, batchSize)

    for (i <- 1 to epochs) {
      sparkNetwork.fit(trainingData)
      sparkNetwork.getNetwork.rnnClearPreviousState() // clear previous state
    }

    val net = sparkNetwork.getNetwork

    //println(sparkNetwork.evaluate(sc.parallelize(List(split.getTest))))

    val locationToSave = new File("./StockPriceLSTM.zip")
    // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)

    for (i <- 1 to 20) {
      val result = net.rnnTimeStep(lastDataSet.getFeatures).getDouble(3)

      println(result)
    }
  }
}
