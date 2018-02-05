package com.dev.bruno.ml.rnn

import org.deeplearning4j.ui.api.UIServer

object NetworkDebuger {

  def main(args: Array[String]): Unit = {

    val uiServer = UIServer.getInstance

    uiServer.enableRemoteListener()
  }
}