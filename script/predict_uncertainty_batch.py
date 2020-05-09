import subprocess
import sys


def main():
    variance_duration = 10  # 10くらいが適切？
    variance_from_date = 1914 - 7 * variance_duration
    variance_to_date = 1914
    interval = int(sys.argv[1])

    # pointwise
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={variance_from_date}', '--local-scheduler'])
    for t in range(variance_from_date, variance_to_date, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise',
             f'--prediction-start-date={variance_from_date}',
             f'--predict-from-date={t}',
             f'--predict-to-date={t + interval}',
             f'--interval={interval}',
             '--local-scheduler'])

    # uncertainty
    subprocess.run(
        ['python', 'main.py', 'm5-forecasting.SubmitUncertainty',
         f'--m5-forecasting.CalculateVariance-interval={interval}',
         f'--m5-forecasting.CalculateVariance-variance-from-date={variance_from_date}',
         f'--m5-forecasting.CalculateVariance-variance-to-date={variance_to_date}',
         '--local-scheduler'])


if __name__ == '__main__':
    main()

# DATA_SIZE=small python script/predict_uncertainty_batch.py 7
# python script/predict_uncertainty_batch.py 7

