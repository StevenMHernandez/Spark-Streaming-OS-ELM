import org.apache.spark._
import breeze.linalg._

object OSELM {
  val L_HIDDEN_NODES = 10

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Indoor Localization with OS-ELM")
    val sc = new SparkContext(conf)

    val trX = loadCSV(sc, "src/main/resources/training_input.csv")
    val trY = loadCSV(sc, "src/main/resources/training_output.csv")
    val teX = loadCSV(sc, "src/main/resources/testing_input.csv")
    val teY = loadCSV(sc, "src/main/resources/testing_output.csv")

    val l = trX.rows
    val dim = trX.cols
    val rangeInput = 0 until l
    val rangeHiddenLayers = 0 until L_HIDDEN_NODES

    val input = trX
    val T = trY

    /**
      *
      * Basic ELM Learning
      *
      */

    def hardLim(a: Double, b: Double, x: Double) = {
      if (a * x + b >= 0) 1 else 0
    }

    var a = DenseVector.rand(L_HIDDEN_NODES) * DenseVector.fill(L_HIDDEN_NODES){2.0} - DenseVector.ones[Double](L_HIDDEN_NODES)
    var bias = DenseVector.rand(L_HIDDEN_NODES) * DenseVector.fill(L_HIDDEN_NODES){2.0} - DenseVector.ones[Double](L_HIDDEN_NODES)

    val H = DenseMatrix.zeros[Double](l, L_HIDDEN_NODES)

    rangeInput.foreach(x => {
      rangeHiddenLayers.foreach(y => {
        H(x,y) = hardLim(a(y), bias(y), input(x,1))
      })
    })

    val P_o = pinv(H.t * H)
    val beta = P_o * H.t * T

    rangeInput.foreach(x => {
      val o = rangeHiddenLayers.map(i => bias(i) * hardLim(a(i), bias(i), input(x, 0))).sum
      val o2 = rangeHiddenLayers.map(i => bias(i) * hardLim(a(i), bias(i), input(x, 1))).sum
      println(o, o2, ";                 ")
    })
  }

  def loadCSV(sc: SparkContext, file: String): DenseMatrix[Double] = {
    val arr = sc.textFile(file).map(line => line.split(",").map(v => v.toDouble)).collect()

    val dim = arr.head.length
    val l = arr.length

    new DenseMatrix(l, dim, arr.flatten)
  }
}
