package com.dev.bruno.ml.nlp

import java.io.File

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover}
import org.apache.spark.sql.SparkSession

object NLP {

  def main(args: Array[String]): Unit = {
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    val conf = new SparkConf()
      .setAppName("NLP")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val dataSet = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", "\t")
      .csv("./data/restaurant_reviews.tsv")

    val vocabSize = 1500

    val Array(train, test) = dataSet.randomSplit(Array(0.8, 0.2), 123)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("Review")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val regexTokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val finisher = new Finisher()
      .setInputCols(Array("token"))
      .setOutputCols(Array("tokens"))
      .setOutputAsArray(true)
      .setIncludeKeys(true)

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("cleantokens")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

    val hashingtf = new HashingTF()
      .setInputCol("cleantokens")
      .setOutputCol("tf")
      .setNumFeatures(vocabSize)

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("features")

    val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.001).setLabelCol("Liked")

    val ovr = new OneVsRest().setClassifier(lr)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        finisher,
        stopWordsRemover,
        hashingtf,
        idf,
        ovr
      ))

    val model = pipeline.fit(train)

    val txTrain = model.transform(train)

    val txTest = model.transform(test)

    val trainDF = txTrain.select("title", "label", "prediction")

    trainDF.show(1000)

    val testDF = txTest.select("title", "label", "prediction")

    testDF.show(1000)

    spark.close()
  }
}