import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(os.path.dirname(__file__))
from models.starnet import StarNet

class TestStarNet(unittest.TestCase):
    def prepare(self):
        pass
    def testParameters(self):
        pass
    def test_train(self):
        pass
    def test_inference(self):
        pass
    def test_validate(self):
        pass

if __name__=='__main__':
    unittest.main()