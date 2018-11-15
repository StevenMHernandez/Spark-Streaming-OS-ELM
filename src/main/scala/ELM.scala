import breeze.linalg.{DenseMatrix, DenseVector, pinv}
import breeze.stats.distributions.Rand
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import scala.collection.mutable

class ELM(override val uid: String) extends Estimator[ELMModel] with ELMTrait {
  var numHiddenNodes = 100

  def this() = this(Identifiable.randomUID("elm"))

  def copy(extra: ParamMap): ELM = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  def datasetToDenseMatrix(dataset: Dataset[Row]): DenseMatrix[Double] = {
    val l = dataset.count().toInt
    val dim = dataset.head()(0).asInstanceOf[mutable.WrappedArray[Double]].size

    val arr = dataset.collect()
      .flatMap(_.toSeq.head.asInstanceOf[mutable.WrappedArray[Double]].toArray.asInstanceOf[Array[Double]])

    new DenseMatrix[Double](l, dim, arr, 0, dim, true)
  }

  override def fit(dataset: Dataset[_]): ELMModel = {
    val input = dataset.select("input")
    val output = dataset.select("output")

    val (inputMatrix, maxX, minX) = normalizeMatrix(datasetToDenseMatrix(input))
    val (outputMatrix, maxY, minY) = normalizeMatrix(datasetToDenseMatrix(output))

    val n = inputMatrix.cols
    val T = outputMatrix

    val a = DenseMatrix.rand(numHiddenNodes, n, Rand.gaussian)
    val bias = DenseVector.rand(numHiddenNodes, Rand.gaussian)

    val H = buildHiddenLayer(a, bias, inputMatrix)
    val K_0 = H.t * H
    val P_0 = pinv(K_0)
    val beta = P_0 * H.t * T

    new ELMModel(a, bias, beta, "sig", maxX, minX, maxY, minY)
  }

  def setHiddenNodes(i: Int): ELM = {
    numHiddenNodes = i
    this
  }
}
