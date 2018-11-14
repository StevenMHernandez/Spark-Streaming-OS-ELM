import org.apache.spark._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.log4j.{Level, Logger}

object OSELM {
  val L_HIDDEN_NODES = 50

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Indoor Localization with OS-ELM")
    val sc = new SparkContext(conf)

    val (trX, maxX, minX) = normalizeMatrix(loadCSV(sc, "src/main/resources/training_input.csv"))
    val (trY, maxY, minY) = normalizeMatrix(loadCSV(sc, "src/main/resources/training_output.csv"))
    val (teX, _, _) = normalizeMatrix(loadCSV(sc, "src/main/resources/testing_input.csv"), maxX, minX)
    val (teY, _, _) = normalizeMatrix(loadCSV(sc, "src/main/resources/testing_output.csv"), maxY, minY)

    val l = trX.rows
    val n = trX.cols
    val m = trY.cols

    val rangeInputs = 0 until l
    val rangeTestInputs = 0 until teX.rows

    val input = trX
    val T = trY

    /**
      *
      * Basic ELM Learning
      *
      */
    val a = DenseMatrix.rand(L_HIDDEN_NODES, n, Rand.gaussian)
    val bias = DenseVector.rand(L_HIDDEN_NODES, Rand.gaussian)

    val H = buildHiddenLayer(a, bias, input)

    val K_0 = H.t * H
    val P_0 = pinv(K_0)
    val beta = P_0 * H.t * T

    val output = H * beta
    val output_testing = buildHiddenLayer(a, bias, teX) * beta

    val predictions = denormalizeMatrix(output_testing, maxY, minY)

    println("====testing-predictions")
    rangeTestInputs.foreach(i => println(predictions(i, 0) + " " + predictions(i, 1)))

    val err_training = sqrt(rangeInputs.map(i => squaredDistance(output(i, ::).t, trY(i, ::).t)).sum)
    val err_testing = sqrt(rangeTestInputs.map(i => squaredDistance(output_testing(i, ::).t, teY(i, ::).t)).sum)
    println("====err-training")
    println(err_training / trY.rows)
    println("====err-testing")
    println(err_testing / teY.rows)
  }

  def loadCSV(sc: SparkContext, file: String): DenseMatrix[Double] = {
    val arr = sc.textFile(file).map(line => line.split(",").map(v => v.toDouble)).collect() // TODO: consider whether this `.collect()` call is necessary

    val dim = arr.head.length
    val l = arr.length

    new DenseMatrix(l, dim, arr.transpose.flatten)
  }

  def denormalizeMatrix(m: DenseMatrix[Double], maxM: DenseVector[Double], minM: DenseVector[Double]): DenseMatrix[Double] = {
    val rangeCol = 0 until m.cols
    val rangeRow = 0 until m.rows

    rangeRow.foreach(i => {
      rangeCol.foreach(j => {
        m(i, j) = (m(i, j) * (maxM(j) - minM(j))) + minM(j) // TODO: consider finding a way to use `.map()` here
      })
    })

    m
  }

  def normalizeMatrix(m: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val maxM = max(m(::, *)).t
    val minM = min(m(::, *)).t

    normalizeMatrix(m, maxM, minM)
  }

  def normalizeMatrix(m: DenseMatrix[Double], maxM: DenseVector[Double], minM: DenseVector[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val rangeCol = 0 until m.cols
    val rangeRow = 0 until m.rows

    rangeRow.foreach(i => {
      rangeCol.foreach(j => {
        m(i, j) = (m(i, j) - minM(j)) / (maxM(j) - minM(j))
      })
    })

    (m, maxM, minM)
  }

  def G(a: DenseVector[Double], b: Double, x: DenseVector[Double]) = {
//    tanh(a.t * x + b)
    sigmoid(a.t * x + b)
//    signum(a.t * x + b)
  }

  def buildHiddenLayer(a: DenseMatrix[Double], bias: DenseVector[Double], input: DenseMatrix[Double]): DenseMatrix[Double] = {
    val l = input.rows
    val num_hidden_nodes = a.rows

    val rangeInputs = 0 until l
    val rangeHiddenLayers = 0 until num_hidden_nodes

    val H = DenseMatrix.zeros[Double](l, num_hidden_nodes)

    rangeInputs.foreach(x => {
      rangeHiddenLayers.foreach(y => {
        H(x, y) = G(a(y, ::).t, bias(y), input(x, ::).t)
      })
    })

    H
  }
}
