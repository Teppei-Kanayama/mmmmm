import gokart
import luigi
import numpy as np
import pandas as pd
import scipy.stats as stats

from m5_forecasting import Submit
from m5_forecasting.data.load import LoadInputData


class PredictUncertainty(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()
    interval: int = luigi.IntParameter()

    def output(self):
        return self.make_target('submission_uncertainty.csv')

    def requires(self):
        accuracy_task = Submit(is_small=self.is_small, interval=self.interval)
        sales_data_task = LoadInputData(filename='sales_train_validation.csv')
        return dict(accuracy=accuracy_task, sales=sales_data_task)

    def run(self):
        accuracy = self.load_data_frame('accuracy')
        sales = self.load_data_frame('sales')
        output = self._run(accuracy, sales)
        self.dump(output)

    @classmethod
    def _run(cls, accuracy, sales):
        sub = accuracy.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id")
        sub["_all_"] = "Total"

        qs = np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
        qs2 = np.log(qs / (1 - qs)) * .065
        ratios = stats.norm.cdf(qs2)
        ratios /= ratios[4]
        ratios = pd.Series(ratios, index=qs)
        ratios.round(3)

        levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
        couples = [("state_id", "item_id"), ("state_id", "dept_id"), ("store_id", "dept_id"), ("state_id", "cat_id"),
                   ("store_id", "cat_id")]
        cols = [f"F{i}" for i in range(1, 29)]

        df = []
        for level in levels:
            df.append(cls.get_group_preds(qs, cols, ratios, sub, level))
        for level1, level2 in couples:
            df.append(cls.get_couple_group_preds(qs, cols, ratios, sub, level1, level2))
        df = pd.concat(df, axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df], axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df.loc[df.index >= len(df.index) // 2, "id"] = df.loc[df.index >= len(df.index) // 2, "id"].str.replace(
            "_validation$", "_evaluation")

        return df

    @staticmethod
    def get_group_preds(qs, cols, ratios, pred, level):
        def quantile_coefs(q):
            return ratios.loc[q].values

        df = pred.groupby(level)[cols].sum()
        q = np.repeat(qs, len(df))
        df = pd.concat([df] * 9, axis=0, sort=False)
        df.reset_index(inplace=True)
        df[cols] *= quantile_coefs(q)[:, None]
        if level != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        df = df[["id"] + list(cols)]
        return df

    @staticmethod
    def get_couple_group_preds(qs, cols, ratios, pred, level1, level2):
        def quantile_coefs(q):
            return ratios.loc[q].values

        df = pred.groupby([level1, level2])[cols].sum()
        q = np.repeat(qs, len(df))
        df = pd.concat([df] * 9, axis=0, sort=False)
        df.reset_index(inplace=True)
        df[cols] *= quantile_coefs(q)[:, None]
        df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1, lev2, q in
                    zip(df[level1].values, df[level2].values, q)]
        df = df[["id"] + list(cols)]
        return df
