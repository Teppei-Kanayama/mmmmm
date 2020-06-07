import unittest
import pandas as pd

from m5_forecasting.adversarial_validation.adversarial_validation import ReshapeAdversarialValidation


class TestReshapeAdversarialValidation(unittest.TestCase):

    def test_run(self) -> None:
        df = pd.DataFrame(dict(
            id=['id1', 'id2', 'id3', 'id1', 'id2', 'id3'],
            start=[117, 117, 117, 482, 482, 482],
            end=[481, 481, 481, 846, 846, 846],
            score=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            ))

        actual = ReshapeAdversarialValidation._run(df)


# id, d, adversarial_score

