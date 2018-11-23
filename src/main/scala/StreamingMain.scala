import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object StreamingMain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("StreamingLinearRegressionExample")
    val ssc = new StreamingContext(conf, Seconds(1))

    val trn = "src/main/resources/training"
    val tst = "src/main/resources/testing"


    val trainingData = ssc.textFileStream(trn).map(SimpleCsvParser.parse)
      .map(x => (
        Seq(x(0), x(1), x(2), x(3), x(4)),
        Seq(x(5), x(6))
      ))
    val testData = ssc.textFileStream(tst).map(SimpleCsvParser.parse)
      .map(x => (
        Seq(x(0), x(1), x(2), x(3), x(4)),
        Seq(x(5), x(6))
      ))

    val model = new StreamingOSELM()
      .setHiddenNodes(5)
      .setActivationFunction(ELM.SIGMOID)

    model.trainOn(trainingData)
    model.predictOn(testData).print()

    ssc.start()
    ssc.awaitTermination()
    ssc.stop()
  }
}
