import argparse
import json


def fixup_old_log(old_log_path, fixed_log_path):
  with open(old_log_path) as f:
    log = json.load(f)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('old_log_path')
  parser.add_argument('fixed_log_path', required=False)

  return parser.parse_args()


def main():
  args = parser.parse_args()

  fixed_log_path = args.fixed_log_path or f'{args.old_log_path}.fixed'
  fixup_old_log(old_log_path, fixed_log_path)


if __name__ == '__main__':
  main()
