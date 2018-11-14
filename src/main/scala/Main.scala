import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("ELM-Example")
      .master("local")
      .getOrCreate()

    val training = spark.read.format("csv")
      .load("src/main/resources/train_XY.csv")

    val testing = spark.read.format("csv")
      .load("src/main/resources/test_XY.csv")

    val elm = new ELM()
      .setHiddenNodes(50)

    val model = elm.fit(training)

    model.transform(testing).show()
  }
}
