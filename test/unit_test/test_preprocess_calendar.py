from datetime import date
import unittest
import pandas as pd
import numpy as np

from m5_forecasting.data.calendar import PreprocessCalendar


class TestPreprocessCalendar(unittest.TestCase):

    def setUp(self) -> None:
        self.calendar = pd.DataFrame(dict(
            date=[date(2011, 1, 29), date(2011, 1, 30), date(2011, 1, 31)],
            wm_yr_wk=[11101, 11101, 11101],
            weekday=['Saturday', 'Sunday', 'Monday'],
            wday=[1, 2, 3],
            month=[1, 1, 1],
            year=['2011', '2011', '2011'],
            d=['d_1', 'd_2', 'd_3'],
            event_name_1=[np.nan, 'SuperBowl', 'ValentinesDay'],
            event_type_1=[np.nan, 'Sporting', 'Religious'],
            event_name_2=[np.nan, np.nan, "Father's day"],
            event_type_2=[np.nan, np.nan, 'Cultural'],
            snap_CA=[0, 0, 1],
            snap_TX=[1, 0, 0],
            snap_WI=[1, 1, 0]
            ))

    def test_run(self):
        actual = PreprocessCalendar._run(self.calendar)
