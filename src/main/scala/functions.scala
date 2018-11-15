import java.util.Date

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import scala.collection.mutable

object functions {
  def getNth(n: Int): UserDefinedFunction = udf { (prediction: Any) =>
    prediction.asInstanceOf[mutable.WrappedArray[Double]](n)
  }

  def time[T](msg: String, block: => T): Unit = {
    val start: Long = new Date().getTime
    block
    val end = new Date().getTime

    val diff = (end - start).toFloat / 1000
    println(s"$msg ${diff}s")
  }
}
