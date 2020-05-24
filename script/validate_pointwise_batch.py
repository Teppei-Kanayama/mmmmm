import subprocess
import sys


def predict_and_validate(train_to_date, validate_from_date, validate_to_date, interval):
    for t in range(validate_from_date, validate_to_date, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise',
             f'--train-to-date={train_to_date}',
             f'--prediction-start-date={validate_from_date}',
             f'--predict-from-date={t}',
             f'--predict-to-date={t + interval}',
             f'--interval={interval}',
             '--local-scheduler'])

    subprocess.run(['python', 'main.py', 'm5-forecasting.ValidatePointwise', f'--interval={interval}',
                    f'--validate-from-date={validate_from_date}', f'--validate-to-date={validate_to_date}',
                    '--local-scheduler'])

def main():
    train_to_date = 1914 - 56  # TODO: 設定ファイルのvalidation期間を0にする

    validate_term1 = [train_to_date, train_to_date + 28]
    validate_term2 = [train_to_date + 28, train_to_date + 56]
    interval = int(sys.argv[1])

    # trian pointwise
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={train_to_date}', '--local-scheduler'])
    predict_and_validate(train_to_date, validate_term1[0], validate_term1[1], interval)
    predict_and_validate(train_to_date, validate_term2[0], validate_term2[1], interval)


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/validate_pointwise_batch.py 7
# python script/validate_pointwise_batch.py 7
