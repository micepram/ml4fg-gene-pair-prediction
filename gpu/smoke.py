"""Compatibility entry point: GPU checks now run inside the shared allowance."""


def main():
    print('GPU checks are included in gpu.run_suite and count toward its three-hour total allowance. '
          'Use the updated Colab_Training.ipynb; no separate GPU training was launched.', flush=True)


if __name__ == '__main__':
    main()
