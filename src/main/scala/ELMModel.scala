import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.Model
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable

class ELMModel(
                val a: DenseMatrix[Double],
                val bias: DenseVector[Double],
                val beta: DenseMatrix[Double],
                val K: DenseMatrix[Double],
                val activationFunction: String,
                val maxX: DenseVector[Double],
                val minX: DenseVector[Double],
                val maxY: DenseVector[Double],
                val minY: DenseVector[Double]
              ) extends Model[ELMModel] with ELMTrait {

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf { features: Any =>
      val arr = features.asInstanceOf[mutable.WrappedArray[Double]].toArray

      val l = 1
      val dim = features.asInstanceOf[mutable.WrappedArray[Double]].size

      val (m, _, _) = normalizeMatrix(new DenseMatrix[Double](l, dim, arr), maxX, minX)

      predict(m)
    }

    dataset.withColumn("predictions", predictUDF(col("input")))
  }

  def transform(dataset: (Seq[Double], Seq[Double])): Seq[Double] = {
    val l = 1
    val dim = dataset._1.size

    val (m, _, _) = normalizeMatrix(new DenseMatrix[Double](l, dim, dataset._1.toArray), maxX, minX)

    predict(m)
  }

  protected def predict(features: DenseMatrix[Double]): Array[Double] = {
    val output = buildHiddenLayer(activationFunction, a, bias, features) * beta
    denormalizeMatrix(output, maxY, minY).toArray
  }

  override def copy(extra: ParamMap): ELMModel = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override val uid: String = Identifiable.randomUID("elm-model")
}
