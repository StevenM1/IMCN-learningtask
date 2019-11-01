import sys
try:
    import exptools2
except:
    sys.append('../../exptools2')
    import exptools2

from LearningSession import LearningSession
from PracticeSession import PracticeSession
import datetime
import os

# # Kill all background processes (macOS only)
# try:
#     import appnope
#
#     appnope.nope()
# except:
#     pass
#
# Set nice to -20: extremely high PID priority. Again unix only
new_nice = -20
sysErr = os.system("sudo renice -n %s %s" % (new_nice, os.getpid()))
if sysErr:
    print('Warning: Failed to renice, probably you arent authorized as superuser')


def main():
    index_number = int(input('What is the pp num? [integer or I will crash]: '))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")

    output_str = f'sub-{index_number}_task-learning_datetime-{timestamp}'
    output_dir = './data'
    settings_file = 'settings.yml'

    sess = PracticeSession(output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=0,
                           index_number=index_number)
    sess.run()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{index_number}_task-learning_datetime-{timestamp}'
    output_dir = './data'
    settings_file = 'settings.yml'

    sess = LearningSession(scanner=True,
                           output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=1,
                           index_number=index_number)
    sess.run()


if __name__ == '__main__':
    main()
