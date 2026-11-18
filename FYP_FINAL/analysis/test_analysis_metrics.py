"""Independent synthetic checks on boundary distances and joins' key outcomes."""
import unittest
import numpy as np
from complete_supplement import surface_scores


class SurfaceTests(unittest.TestCase):
    def test_identical_and_empty(self):
        a=np.zeros((30,30),bool);a[8:18,8:18]=True
        r=surface_scores(a,a)
        self.assertEqual(r['surface_hd95_px'],0)
        self.assertEqual(r['boundary_f1'],1)
        r=surface_scores(np.zeros_like(a),a)
        self.assertTrue(np.isnan(r['surface_hd95_px']))
        self.assertEqual(r['boundary_f1'],0)

    def test_translated_square(self):
        a=np.zeros((30,30),bool);a[8:18,8:18]=True
        b=np.zeros_like(a);b[8:18,9:19]=True
        r=surface_scores(a,b)
        self.assertAlmostEqual(r['surface_hd95_px'],1)
        self.assertEqual(r['boundary_f1'],1)


if __name__=='__main__':unittest.main()
