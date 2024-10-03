import os
from abc import ABC, abstractmethod, abstractproperty
from contextlib import closing
from typing import Dict

from operator import TrinoOperator
from lightgbm import Dataset
from prettytable import PrettyTable
import pandas as pd
from loguru import logger


class BasicDataSet(ABC):
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.template = os.path.join(dirname, "template", self.template_name)

    def get_lgbm_dataset(self):
        _label = self.basic_information["label"]
        raw = self.fetch()
        _cat_columns = raw.select_dtypes("object").columns

        for c in _cat_columns:
            raw[c] = raw[c].fillna("").astype("category")

        dataset = Dataset(
            raw.drop(_label, axis=1),
            label=raw[_label],
            categorical_feature=_cat_columns.to_list(),
            free_raw_data=False,
        )
        dataset.construct()
        return dataset

    def get_pandas(self):
        return self.fetch()

    def execute(self):
        trino = TrinoOperator()
        trino.execute_file(self.template, **self.args)

    def fetch(self):
        self.execute()
        _target_table = self.basic_information["target_table"]
        if _target_table:
            trino = TrinoOperator()
            return trino.execute_pandas(f"""select * from {_target_table}""")

    def set_args(self, **kwargs):
        self.args.update(kwargs)

    def show_template(self):
        with closing(open(self.template, "r")) as f:
            return f.read()

    def description(self):
        _ = PrettyTable(field_names=["arg name", "type", "description"])
        for f in self.args_list:
            _.add_row(f)

        return _.get_string()

    def template_name(self):
        return None

    def args_list(self):
        return None

    def basic_information(self):
        return None


class DataSetStack(BasicDataSet):
    def __init__(self, *dataset_list):
        self.dl = dataset_list
        self.covar_list = []

    def add_covar(self, covar):
        assert len(covar) == len(self.dl), f"协变量数量应与数据集数量一致"
        self.covar_list.append(covar)

    def get_pandas(self):
        res = [dataset.fetch() for dataset in self.dl]

        for covar_index in range(len(self.covar_list)):
            for dataset_index in range(len(self.dl)):

                res[dataset_index][f"bizmodel_covar_{covar_index}"] = self.covar_list[
                    covar_index
                ][dataset_index]

        return pd.concat(res, axis=0, ignore_index=True)

    def get_lgbm_dataset(self, is_train: bool = True):
        if not hasattr(self, "raw"):
            self.raw = self.get_pandas()
        raw = self.raw

        _label = self.dl[0].basic_information["label"]
        _cat_columns = raw.drop(_label, axis=1).select_dtypes("object").columns

        for c in _cat_columns:
            raw[c] = raw[c].fillna("").astype("category")

        if is_train:
            dataset = Dataset(
                raw.drop(_label, axis=1),
                label=raw[_label],
                categorical_feature=_cat_columns.to_list(),
                free_raw_data=False,
            )
        else:
            dataset = Dataset(
                raw.drop(_label, axis=1),
                categorical_feature=_cat_columns.to_list(),
                free_raw_data=False,
            )
        dataset.construct()
        return dataset


class TimeseriesDataset(BasicDataSet):
    pass
