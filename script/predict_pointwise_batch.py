import subprocess
import sys


def main():
    train_to_date = 1914  # TODO: this would be 1942 on 2nd stage
    predict_to_date = 1942  # TODO: this would be 1970 on 2nd stage
    interval = int(sys.argv[1])

    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={train_to_date}',
                    '--local-scheduler'])
    for t in range(train_to_date, predict_to_date, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise',
             f'--train-to-date={train_to_date}',
             f'--prediction-start-date={train_to_date}',
             f'--predict-from-date={t}',
             f'--predict-to-date={t + interval}',
             f'--interval={interval}',
             '--local-scheduler'])
    subprocess.run(['python', 'main.py', 'm5-forecasting.SubmitPointwise', f'--interval={interval}', '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/predict_pointwise_batch.py 7
# python script/predict_pointwise_batch.py 1
