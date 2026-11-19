import unittest

import numpy as np

from ccs_fd import Predictor
from ccs_fd.features import extract_features


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(device="cpu")

    def test_window_and_features_agree(self):
        rng = np.random.default_rng(7)
        windows = rng.standard_normal((3, 2048, 18)).astype(np.float32)
        result = self.predictor.predict_windows(windows, return_detail=True)
        features = extract_features(windows)
        np.testing.assert_array_equal(self.predictor.predict(features), result["pred"])
        self.assertEqual(features.shape, (3, 2628))
        pack = self.predictor.pack
        for candidate, pred, distance in zip(result["label"], result["pred"], result["distance"]):
            state = "".join(c for c in candidate if c in "CDE") or "H"
            cutoff = pack["thresholds"][pack["hypotheses"].index(state)]
            self.assertEqual(pred, candidate if distance <= cutoff else "UNKNOWN")

    def test_empty_and_incompatible_windows(self):
        self.assertEqual(self.predictor.predict_windows(np.empty((0, 2048, 18))).shape, (0,))
        with self.assertRaises(ValueError):
            self.predictor.predict_windows(np.zeros((1, 2048, 1)))


if __name__ == "__main__":
    unittest.main()
