import unittest
from ecg_processor import analyze_ecg

class TestECG(unittest.TestCase):
    def test_ecg_analysis(self):
        result = analyze_ecg()
        self.assertEqual(type(result), str)

if __name__ == '__main__':
    unittest.main()