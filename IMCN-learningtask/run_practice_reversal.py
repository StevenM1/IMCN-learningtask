import sys, os
import yaml
try:
    import exptools2
except:
    sys.append('../../exptools2')
    import exptools2

from PracticeSession import *
import datetime
from run_learning_SAT import input_must_be, input_int


def run_practice(index_number, SAT_first, practice_n, dir):

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{index_number}_task-learning-practice_datetime-{timestamp}'
    output_dir = os.path.join(dir, 'data_practice-learning')

    settings_template = os.path.join(dir, 'default_settings_practice.yml')
    settings_fn = os.path.join(dir, 'tmp_settings_directory',
                               f'sub-{index_number}_task-learning-practice_settings.yml')

    # check dirs, make them if not present
    for dir_name in ['tmp_settings_directory', output_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    with open(settings_template, 'r') as stream:
        settings = yaml.load(stream)
    settings['stimulus']['font'] = 'Agathodaimon'
    settings['stimulus']['font_fn'] = os.path.join(dir, 'lib', 'AGATHODA.TTF')
    with open(settings_fn, 'w') as f:
        yaml.dump(settings, f)

    sess = PracticeSession(output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_fn,
                           start_block=0,
                           SAT_first=SAT_first,
                           practice_n=practice_n
                           )
    sess.run()


def main():
    index_number = input_int('What is the subject num? [integer]: ')
    SAT_first = False #input_must_be('Is/was SAT first in this session? [y, n] ', options=('y', 'n')) == 'y'
    practice_n = 1 #input_int('Is this the first / second RL practice of this session? [1/2]: ')

    run_practice(index_number, SAT_first, practice_n, dir='./')



if __name__ == '__main__':
    main()
