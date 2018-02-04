package com.dev.bruno.ml.rnn

import java.io.File

import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.FileStatsStorage

object NetworkDebuger {

  def main(args: Array[String]): Unit = {

    val statsStorage = new FileStatsStorage(new File("./training_stats.dl4j"))

    val uiServer = UIServer.getInstance

    uiServer.attach(statsStorage)

    //uiServer.enableRemoteListener()
  }
}
