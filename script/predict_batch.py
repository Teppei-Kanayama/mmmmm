import subprocess


def main():
    subprocess.run(['python', 'main.py', 'm5-forecasting.Train', '--local-scheduler'])
    interval = 7
    for t in range(1914, 1942, interval):
        subprocess.run(['python', 'main.py', 'm5-forecasting.Predict', f'--from-date={t}', f'--to-date={t+interval}',
                        f'--interval={interval}', '--local-scheduler'])
    subprocess.run(['python', 'main.py', 'm5-forecasting.Submit', f'--interval={interval}', '--local-scheduler'])


if __name__ == '__main__':
    main()
