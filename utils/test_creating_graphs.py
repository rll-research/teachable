import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MyTestCase(unittest.TestCase):


    def test_something(self):
        grounding = pd.read_csv("/home/claire/Projects_Test/teachable-rl/scripts/logs/persisted_models_distill/claire/results.csv")
        mlp_distill = pd.read_csv("/home/claire/Projects_Test/teachable-rl/scripts/logs/persisted_models_distill/distilled_with_mlp_2/results.csv")
        paper_distill = pd.read_csv("/home/claire/Projects_Test/teachable-rl/scripts/logs/persisted_models_distill/original_distillation/results.csv")

        mlp_distill = pd.concat([grounding, mlp_distill])
        paper_distill = pd.concat([grounding, paper_distill])

        plt.plot(range(0, paper_distill.shape[0]*20, 20), paper_distill['reward'], label="original distillation")
        plt.plot(range(0, mlp_distill.shape[0]*20, 20), mlp_distill['reward'], label="our version")
        plt.xlabel("# iterations")
        plt.ylabel("reward")
        plt.title("reward over time for different distillation methods")
        plt.legend()
        plt.show()
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
