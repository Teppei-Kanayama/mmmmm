import subprocess
import sys


def main():
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', '--local-scheduler'])

    interval = int(sys.argv[1])
    for t in range(1914, 1942, interval):
        subprocess.run(['python', 'main.py', 'm5-forecasting.PredictPointwise', f'--predict-from-date={t}', f'--predict-to-date={t+interval}',
                        f'--interval={interval}', '--local-scheduler'])
    subprocess.run(['python', 'main.py', 'm5-forecasting.SubmitPointwise', f'--interval={interval}', '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/predict_pointwise_batch.py 7
# python script/predict_pointwise_batch.py 1

# DATA_SIZE=small python script/predict_pointwise_batch.py 7

