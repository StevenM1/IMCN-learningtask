import sys
try:
    import exptools2
except:
    sys.path.append('../exptools2')
    import exptools2

from LearningSession import *
import datetime

# Kill all background processes (macOS only)
try:
    import appnope
    appnope.nope()
except:
    pass

# Set nice to -20: extremely high PID priority. Again unix only
new_nice = -20
sysErr = os.system("sudo renice -n %s %s" % (new_nice, os.getpid()))
if sysErr:
    print('Warning: Failed to renice, probably you arent authorized as superuser')


def main():

    index_number = int(input('What is the pp num? [integer or I will crash]: '))
    start_block = input('Start block? [default 1]: ')
    try:
        start_block = int(start_block)
    except:
        start_block = 1

    if start_block > 1:
        pass
        # ToDo: find last run data to get points from

    scanner = input('Are you in the scanner? [y/n, default n]: ') or 'n'
    while not scanner in ['n', 'y']:
        print('I don''t understand that. Please enter ''y'' or ''n''.')
        scanner = input('Are you in the scanner? [y/n, default n]: ') or 'n'

    simulate = 'n'
    if scanner == 'n':
        simulate = input('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'
        while not simulate in ['n', 'y']:
            print('I don''t understand that. Please enter ''y'' or ''n''.')
            simulate = input('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{index_number}_task-learning_datetime-{timestamp}'
    output_dir = './data'
    if simulate == 'y':
        settings_file = 'settings_simulate.yml'
    else:
        settings_file = 'settings.yml'

    sess = LearningSession(scanner=scanner,
                           output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=start_block,
                           index_number=index_number)
    sess.run()


if __name__ == '__main__':
    main()