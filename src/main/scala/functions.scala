import java.util.Date

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import scala.collection.mutable

object functions {
  def getNth(n: Int): UserDefinedFunction = udf { prediction: Any =>
    prediction.asInstanceOf[mutable.WrappedArray[Double]](n)
  }

  def time[T](msg: String, block: => T): Unit = {
    val start: Long = new Date().getTime
    block
    val end = new Date().getTime

    val diff = (end - start).toFloat / 1000
    println(s"$msg ${diff}s")
  }

  def evaluate(predictions: DataFrame): Unit = {
    val df = predictions.select("actualX", "predictionX").rdd
      .map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

    val metric = new RegressionMetrics(df, false)

    println("explainedVariance", metric.explainedVariance)
    println("meanAbsoluteError", metric.meanAbsoluteError)
    println("meanSquaredError", metric.meanSquaredError)
    println("r2", metric.r2)
    println("rootMeanSquaredError", metric.rootMeanSquaredError)
  }
}
