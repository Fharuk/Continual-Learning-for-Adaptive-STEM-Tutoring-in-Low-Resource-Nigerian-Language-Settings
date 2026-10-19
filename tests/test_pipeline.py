import unittest
from src.metrics import adaptation_speed, backward_transfer, forward_transfer
from src.simulate import simulate
from src.experiment import run

class PipelineTests(unittest.TestCase):
    def test_simulator_reproducible_and_marked(self):
        a = simulate(2, 3, 42)
        b = simulate(2, 3, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2 * 6 * 3)
        self.assertTrue(all(r["data_status"] == "synthetic_not_real_student_data" for r in a))

    def test_bwt(self):
        self.assertAlmostEqual(backward_transfer([[.6, None], [.4, .7]]), -.2)

    def test_fwt(self):
        self.assertAlmostEqual(forward_transfer([[.6, .4], [.5, .7]], [.5, .2]), .2)

    def test_adaptation(self):
        self.assertEqual(adaptation_speed([.3, .5, .7], .6), 3)
        self.assertIsNone(adaptation_speed([.3, .5], .8))

    def test_experiment_output_shape(self):
        result = run(7, n_students=10, qpt=3, epochs=1)
        self.assertEqual(result["rows"], 10 * 6 * 3)
        self.assertEqual(set(result["metrics"]), {"Fine-tuning", "EWC", "Replay", "Joint training", "Task-isolated models"})
        self.assertTrue(all(0.0 <= x["accuracy"] <= 1.0 for x in result["metrics"].values()))

if __name__ == "__main__":
    unittest.main()
