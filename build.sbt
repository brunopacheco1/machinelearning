name := "machinelearning"

version := "1.0"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.1"

libraryDependencies += "org.deeplearning4j" %% "dl4j-spark" % "0.9.1_spark_2"