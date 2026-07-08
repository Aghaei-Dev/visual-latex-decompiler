import sys
import argparse


def process_args(args):
    parser = argparse.ArgumentParser(
        description='Evaluate exact-match accuracy.')

    parser.add_argument('--target-formulas', dest='target_file',
                        type=str, required=True,
                        help=(
                            'target formulas file'
                        ))

    parser.add_argument('--predicted-formulas', dest='predicted_file',
                        type=str, required=True,
                        help=(
                            'predicted formulas file'
                        ))

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)

    target_formulas_file = parameters.target_file
    predicted_formulas_file = parameters.predicted_file

    target_formulas = open(target_formulas_file).readlines()
    predicted_formulas = open(predicted_formulas_file).readlines()

    i = 0

    total_match = 0
    if len(target_formulas) != len(predicted_formulas):
        print("number of formulas doesn't match")
        return
    n = len(target_formulas)
    for tf, pf in zip(target_formulas, predicted_formulas):
        i += 1
        if i % 2000 == 0:
            print("{}/{}".format(i, n))

        # token-level exact match -- the whole formula has to be identical
        true_token = tf.strip().split(' ')
        predicted_tokens = pf.strip().split(' ')
        if true_token == predicted_tokens:
            total_match += 1
    print("{}/{}".format(n, n))
    print('Exact Match Accuracy: %f' % (float(total_match) / n))


if __name__ == '__main__':
    main(sys.argv[1:])
