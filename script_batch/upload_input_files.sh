#!/usr/bin/env bash

set -euC

aws s3 cp resources/input/calendar.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/sales_train_validation.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/sample_submission.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/sample_submission_uncertainty.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/sell_prices.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/weights_validation.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/sell_prices.csv s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/grid_part_1.pkl s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/grid_part_2.pkl s3://kaggle-m5-filestore/input/
aws s3 cp resources/input/grid_part_3.pkl s3://kaggle-m5-filestore/input/

