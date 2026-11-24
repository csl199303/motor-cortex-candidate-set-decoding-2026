"""Small deterministic tests for held-out prediction invariants."""

import unittest

import numpy as np

from run_gnode_within_set_decoding import fixed_folds
from run_iscience_decoding import DIRECTIONS, decode, evaluate_session


class DecoderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9312)
        self.y = np.tile(DIRECTIONS, 9)
        self.x = rng.poisson(2, size=(len(self.y), 3, 4)).astype(float)
        self.folds = fixed_folds(self.y)

    def test_outer_test_rows_cannot_affect_each_others_predictions(self):
        for f in range(5):
            a, b = np.flatnonzero(self.folds == f)[:2]
            original, temperature = decode(self.x, self.y, self.folds, DIRECTIONS)
            changed_x, changed_y = self.x.copy(), self.y.copy()
            changed_x[a] = 1e5
            changed_y[a] = changed_y[a] % 6 + 1
            result, perturbed = decode(changed_x, changed_y, self.folds, DIRECTIONS)
            np.testing.assert_allclose(original[b], result[b], atol=1e-12)
            np.testing.assert_array_equal(temperature[f], perturbed[f])

    def test_constant_activity_recovers_cue_priors(self):
        for condition in (2, 3):
            score, h = evaluate_session(np.ones_like(self.x), self.y, condition)
            np.testing.assert_allclose(score[:, :4], np.repeat(
                [[0, np.log2(6 / condition), np.log2(6 / condition), 0]],
                self.x.shape[-1], axis=0,
            ), atol=1e-12)
            np.testing.assert_allclose(np.exp(h[3]), np.repeat(
                h[4][:, :, None], self.x.shape[-1], axis=2,
            ))
            np.testing.assert_allclose(score[:, 5:], np.repeat(
                [[1 / 6, 1 / condition, 1 / condition]], self.x.shape[-1], axis=0,
            ))


if __name__ == "__main__":
    unittest.main()
