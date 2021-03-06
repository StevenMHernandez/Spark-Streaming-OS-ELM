name := "sparkIndoorLocalization"

version := "0.1"

scalaVersion := "2.11.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.2"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.3.2"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.2"

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2"

//  // The visualization library is distributed separately as well.
//  // It depends on LGPL code
//  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.3.1_0.10.0" % "test"

parallelExecution in Test := false

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"