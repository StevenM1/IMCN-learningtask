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


def main():
    task_name = get_task_name()
    index_number = input_int('What is the subject num? [integer]: ')
    session_n = input_must_be('What is the session number? [1 or 2] ', options=('1', '2'))
    age = input_int('What is your age? [integer] ')
    gender = input_must_be('What is your gender? [f/m/NA] ', options=('f', 'm', 'NA'))
    start_block = input('Start block? [default 1]: ')
    try:
        start_block = int(start_block)
    except:
        start_block = 1

    # Inputs end above
    # set-up input file names
    subject_str = str(index_number).zfill(3)
    settings_template = './default_settings.yml'
    settings_fn = f'./tmp_settings_directory/sub-{subject_str}_ses-{session_n}_task' \
                  f'-{task_name}_settings.yml'
    design_fn = f'./designs_{task_name}/sub-{subject_str}_ses-{session_n}_task-{task_name}_design.csv'

    # set-up output file names
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{subject_str}_ses-{session_n}_task-{task_name}_datetime-{timestamp}'
    output_dir = f'./data_{task_name}'

    # If this task is the SAT version, show timing feedback. If not, don't show this.
    show_timing_feedback = task_name == 'SAT-learning'  # create bool

    # Overwrite the "stimulus/font" and "stimulus/font_fn" attributes in the .yml-file
    # NB: This assumes that session 1 *always* uses Agathodaimon, session 2 *always* uses Glagolitic
    use_font = 'Agathodaimon' if session_n == '1' else 'Glagolitsa'
    use_font_fn = './lib/Glagolitsa.ttf' if session_n == '1' else './lib/Glagolitsa.ttf'
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
                           scanner=False,
                           run_scanner_design=False)
    sess.run()


if __name__ == '__main__':
    main()