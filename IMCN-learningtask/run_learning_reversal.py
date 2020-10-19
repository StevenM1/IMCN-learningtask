import sys
import yaml
try:
    import exptools2
except:
    sys.path.append('../exptools2')
    import exptools2

from LearningSession import *
import datetime


def input_int(q):
    response = ''
    while not isinstance(response, int):
        response = input(q)
        try:
            response = int(response)
        except:
            print('Please enter an integer. {}'.format(q))
    return response


def input_must_be(q, options=('f', 'm', 'NA')):
    response = ''
    while response not in options:
        response = input(q)
        if response not in options:
            print('Please enter one of {}. {}'.format(options, q))
    return response


def get_task_name():
    response = input_must_be('Which task do you want to run? [enter "0" for SAT, "1" for Reversal learning] ',
                             options=('0', '1'))
    if response == '0':
        return 'SAT-learning'
    elif response == '1':
        return 'reversal-learning'


def run_learning(task_name, index_number, age, gender, start_block, dir='.'):
    # set-up input file names
    subject_str = str(index_number).zfill(3)
    settings_template = os.path.join(dir, 'default_settings.yml')
    settings_fn = os.path.join(dir, 'tmp_settings_directory',
                               f'sub-{subject_str}_task-{task_name}_settings.yml')
    design_fn = os.path.join(dir,
                             f'designs_{task_name}_TR-1p38',
                             f'sub-{subject_str}_task-{task_name}_design.csv')

    # set-up output file names
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_str = f'sub-{subject_str}_task-{task_name}_datetime-{timestamp}'
    output_dir = os.path.join(dir, f'data_{task_name}')

    # If this task is the SAT version, show timing feedback. If not, don't show this.
    show_timing_feedback = task_name == 'SAT-learning'  # create bool

    # Overwrite the "stimulus/font" and "stimulus/font_fn" attributes in the .yml-file
    use_font = 'Agathodaimon'
    use_font_fn = os.path.join(dir, 'lib', 'AGATHODA.TTF')

    # check dirs, make them if not present
    for dir_name in ['tmp_settings_directory', output_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    with open(settings_template, 'r') as stream:
        settings = yaml.load(stream)
    settings['stimulus']['font'] = use_font
    settings['stimulus']['font_fn'] = use_font_fn
    with open(settings_fn, 'w') as f:
        yaml.dump(settings, f)

    # set-up session
    sess = LearningSession(output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_fn,
                           design_fn=design_fn,
                           gender=gender,
                           age=age,
                           show_timing_feedback=show_timing_feedback,
                           start_block=start_block,
                           scanner=True,
                           run_scanner_design=True)
    sess.run()


def main():
    task_name = 'reversal-learning' #get_task_name()
    index_number = input_int('What is the subject num? [integer]: ')
    age = input_int('What is your age? [integer] ')
    gender = input_must_be('What is your gender? [f/m/NA] ', options=('f', 'm', 'NA'))
    start_block = input('Start block? [default 1]: ')
    try:
        start_block = int(start_block)
    except:
        start_block = 1

    run_learning(task_name=task_name, index_number=index_number,
                 age=age, gender=gender, start_block=start_block)


if __name__ == '__main__':
    main()
