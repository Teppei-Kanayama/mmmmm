import subprocess


def main():
    for t in range(1914, 1942, 1):
        subprocess.run(['python', 'main.py', 'm5-forecasting.Predict', f'--from-date={t}', f'--to-date={t+1}',
                        '--interval 1', '--local-scheduler'])


if __name__ == '__main__':
    main()
