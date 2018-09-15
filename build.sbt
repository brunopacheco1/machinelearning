name := "machinelearning"

version := "1.0"

scalaVersion := "2.11.12"

classpathTypes += "maven-plugin"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"

libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.1.0"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"

libraryDependencies += "org.deeplearning4j" %% "dl4j-spark-ml" % "0.9.1_spark_2"

libraryDependencies += "org.deeplearning4j" %% "deeplearning4j-ui" % "0.9.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1"

libraryDependencies += "co.theasi" %% "plotly" % "0.2.0"

excludeDependencies ++= Seq(
  ExclusionRule("ch.qos.logback","logback-classic")
)