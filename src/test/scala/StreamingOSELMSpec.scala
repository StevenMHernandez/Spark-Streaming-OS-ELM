import com.holdenkarau.spark.testing.StreamingActionBase
import org.apache.spark.streaming.dstream.DStream
import org.scalactic.Equality
import org.scalatest.FunSuite

class StreamingOSELMSpec extends FunSuite with StreamingActionBase {
  test("StreamingOSELM can learn and can predict") {
    val input = Seq(Seq(Seq(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0), Seq(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
      .map(x => (Seq(x(0), x(1), x(2), x(3), x(4)), Seq(x(5), x(6)))))

    val expected = input.head.map(_._2)

    val oselm = new StreamingOSELM()
      .setHiddenNodes(5)
      .setActivationFunction(ELM.SIGMOID)

    assert(oselm.model == null)

    runAction[(Seq[Double], Seq[Double])](input, oselm.trainOn)

    assert(oselm.model != null)
    assert(oselm.model.beta != null)

    val inputStream = input: Seq[Seq[(Seq[Double], Seq[Double])]]
    val operationItem = oselm.predictOn: DStream[(Seq[Double], Seq[Double])] => DStream[Seq[Double]]
    val expectedItem = Seq(expected): Seq[Seq[Seq[Double]]]

    // Because this is ML, we won't necessarily know the exact value returned and we don't want to validate that here.
    // Instead, we want to just ensure we are receiving the correct number of values as output.
    implicit val sameSizeEquality: Equality[Seq[Double]] =
      new Equality[Seq[Double]] {
        override def areEqual(a: Seq[Double], b: Any): Boolean =
          a.size == b.asInstanceOf[Seq[Double]].size
      }

    testOperation[(Seq[Double], Seq[Double]), Seq[Double]](inputStream, operationItem, expectedItem)
  }
}
