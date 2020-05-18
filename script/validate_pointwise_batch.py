import subprocess
import sys


def main():
    validate_from_date = 1914 - 56  # TODO: 設定ファイルのvalidation期間を0にする
    validate_to_date = 1914
    interval = int(sys.argv[1])

    # # pointwise
    # subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={validate_from_date}', '--local-scheduler'])
    # for t in range(validate_from_date, validate_to_date, interval):
    #     subprocess.run(
    #         ['python', 'main.py', 'm5-forecasting.PredictPointwise',
    #          f'--prediction-start-date={validate_from_date}',
    #          f'--predict-from-date={t}',
    #          f'--predict-to-date={t + interval}',
    #          f'--interval={interval}',
    #          '--local-scheduler'])

    # validate
    subprocess.run(
        ['python', 'main.py', 'm5-forecasting.ValidatePointwise',
         f'--interval={interval}',
         f'--validate-from-date={validate_from_date}',
         f'--validate-to-date={validate_to_date}',
         '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/validate_pointwise_batch.py 7
# python script/validate_pointwise_batch.py 7
