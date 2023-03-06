import os
import argparse
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/wikihow')
    parser.add_argument('--save_metric', action='store_false')
    parser.add_argument('--split', default='test')
    opt = parser.parse_args()

    weights_bases = [
        '../runs/baseline',
    ]


    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_metric:  # Save the final metric for the current best model
            save_path = os.path.join(base, '{}_metric.json'.format(opt.split))
        else:
            save_path = None
        evaluation.eval(model_path, data_path=opt.data_path, split=opt.split, save_path=save_path)


if __name__ == '__main__':
    main()