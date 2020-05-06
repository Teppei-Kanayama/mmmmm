import pandas as pd
from gokart.file_processor import CsvFileProcessor


class RoughCsvFileProcessor(CsvFileProcessor):
    def dump(self, obj, file):
        assert isinstance(obj, (pd.DataFrame, pd.Series)), \
            f'requires pd.DataFrame or pd.Series, but {type(obj)} is passed.'
        obj.to_csv(file, index=False, sep=self._sep, header=True, float_format='%.3g')