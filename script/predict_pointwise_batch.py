import subprocess
import sys


def predict_and_validate(train_to_date, predict_from_date, predict_to_date, interval):
    for t in range(predict_from_date, predict_to_date, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise',
             f'--train-to-date={train_to_date}',
             f'--prediction-start-date={predict_from_date}',
             f'--predict-from-date={t}',
             f'--predict-to-date={t + interval}',
             f'--interval={interval}',
             '--local-scheduler'])


def main():
    interval = int(sys.argv[1])

    # train
    train_to_date = 1914  # TODO: 1942
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={train_to_date}',
                    '--local-scheduler'])

    # predict
    predict_and_validate(train_to_date=train_to_date, predict_from_date=1914, predict_to_date=1942, interval=interval)
    predict_and_validate(train_to_date=train_to_date, predict_from_date=1942, predict_to_date=1970, interval=interval)

    # submit
    subprocess.run(['python', 'main.py', 'm5-forecasting.SubmitPointwise', f'--interval={interval}', '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/predict_pointwise_batch.py 7
# python script/predict_pointwise_batch.py 7
