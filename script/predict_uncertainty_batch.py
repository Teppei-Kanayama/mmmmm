import subprocess
import sys


def main():
    # pointwise
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', '--local-scheduler'])
    interval = int(sys.argv[1])
    for t in range(1914, 1942, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise', f'--from-date={t}', f'--to-date={t + interval}',
             f'--interval={interval}', '--local-scheduler'])
    subprocess.run(
        ['python', 'main.py', 'm5-forecasting.SubmitPointwise', f'--interval={interval}', '--local-scheduler'])

    # uncertainty
    subprocess.run(['python', 'main.py', 'm5-forecasting.MakeGroundTruth', f'--interval={interval}', '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/predict_uncertainty_batch.py 7

