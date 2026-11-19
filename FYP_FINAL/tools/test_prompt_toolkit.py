"""Geometry and sampling invariants for the reusable prompt interface."""
import unittest
import numpy as np
from prompt_toolkit import box_from_mask, generate_prompts, shift_box


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.m = np.zeros((13, 17), bool)
        self.m[2:8, 3:11] = True

    def test_historical_and_half_open_endpoints(self):
        self.assertEqual(box_from_mask(self.m, 0), [3, 2, 10, 7])
        self.assertEqual(box_from_mask(self.m, 0, 'half_open'), [3, 2, 11, 8])
        self.assertEqual(box_from_mask(self.m, 30), [0, 0, 17, 13])

    def test_empty_and_full_masks(self):
        self.assertEqual(generate_prompts(np.zeros((4, 4)))['status'], 'empty_mask')
        full = generate_prompts(np.ones((4, 4)))
        self.assertTrue(all(p['label'] == 1 for p in full['points']))

    def test_reproducible_valid_points(self):
        a = generate_prompts(self.m, 77)
        self.assertEqual(a, generate_prompts(self.m, 77))
        for p in a['points']:
            x, y = p['xy']
            self.assertEqual(int(self.m[y, x]), p['label'])

    def test_clipping_and_nested_expansions(self):
        b = [box_from_mask(self.m, e) for e in [0, 1, 3, 30]]
        for a, c in zip(b, b[1:]):
            self.assertTrue(c[0] <= a[0] and c[1] <= a[1] and c[2] >= a[2] and c[3] >= a[3])
        self.assertEqual(shift_box(b[0], self.m.shape, 100, -100), [17., 0., 17., 0.])

    def test_invalid_input_and_single_pixel(self):
        with self.assertRaises(ValueError): box_from_mask(self.m, -1)
        with self.assertRaises(ValueError): box_from_mask(np.ones((2, 2, 2)))
        one = np.zeros((3, 3), bool); one[1, 1] = True
        self.assertEqual(box_from_mask(one, 0), [1, 1, 1, 1])
        self.assertEqual(box_from_mask(one, 0, 'half_open'), [1, 1, 2, 2])


if __name__ == '__main__': unittest.main()
