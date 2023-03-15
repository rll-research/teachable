import unittest
import matplotlib.pyplot as plt
import pandas as pd

class MyTestCase(unittest.TestCase):


    def test_something(self):
        data = pd.read_csv("/Users/1000ber-5078/PycharmProjects/teachable-rl/scripts/logs/persisted_models_distill/claire/results.csv")


        plt.plot(data['itr'], data['reward'])
        plt.show()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
