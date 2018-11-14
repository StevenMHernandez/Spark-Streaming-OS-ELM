import breeze.linalg.{DenseMatrix, DenseVector, pinv}
import breeze.stats.distributions.Rand
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

class ELM(override val uid: String) extends Estimator[ELMModel] with ELMTraits {
  var numHiddenNodes = 50

  def this() = this(Identifiable.randomUID("elm"))

  def copy(extra: ParamMap): ELM = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  def makeDenseMatrix(l: Int, dim: Int, x: Array[Row]): DenseMatrix[Double] = {
    new DenseMatrix[Double](l, dim, x.map(_.toSeq.toArray).flatMap(x => x).map(x => x.asInstanceOf[String].toDouble), 0, dim, true)
  }

  override def fit(dataset: Dataset[_]): ELMModel = {
    val input = dataset.select("_c0", "_c1", "_c2", "_c3", "_c4").collect()
    val output = dataset.select("_c5", "_c6").collect()

    val l = input.length
    val dimX = input.head.length
    val dimY = output.head.length

    val (inputMatrix, maxX, minX) = normalizeMatrix(makeDenseMatrix(l, dimX, input))
    val (outputMatrix, maxY, minY) = normalizeMatrix(makeDenseMatrix(l, dimY, output))

    val n = dimX
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
