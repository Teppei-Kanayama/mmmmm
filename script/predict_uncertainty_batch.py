import subprocess
import sys


def run_pointwise_model(train_to_date, validate_from_date, validate_to_date, interval):
    subprocess.run(['python', 'main.py', 'm5-forecasting.TrainPointwiseModel', f'--train-to-date={train_to_date}', '--local-scheduler'])
    for t in range(validate_from_date, validate_to_date, interval):
        subprocess.run(
            ['python', 'main.py', 'm5-forecasting.PredictPointwise',
             f'--train-to-date={train_to_date}',
             f'--prediction-start-date={validate_from_date}',
             f'--predict-from-date={t}',
             f'--predict-to-date={t + interval}',
             f'--interval={interval}',
             '--local-scheduler'])


def main():
    variance_duration = 10
    variance_to_date = 1914  # TODO: 1942
    variance_from_date = variance_to_date - 7 * variance_duration
    interval = int(sys.argv[1])

    run_pointwise_model(train_to_date=variance_from_date,
                        validate_from_date=variance_from_date,
                        validate_to_date=variance_to_date,
                        interval=interval)

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
