from LearningSession import *
from PracticeSession import PracticeSession
import datetime
import sys


# Kill all background processes (macOS only)
try:
    import appnope
    appnope.nope()
except:
    pass

# try:
#     # Kill Finder during execution (this will be fun)
#     applescript="\'tell application \"Finder\" to quit\'"
#     shellCmd = 'osascript -e '+ applescript
#     os.system(shellCmd)
# except:
#     pass

# Set nice to -20: extremely high PID priority
new_nice = -20
sysErr = os.system("sudo renice -n %s %s" % (new_nice, os.getpid()))
if sysErr:
    print('Warning: Failed to renice, probably you arent authorized as superuser')


def main():

    # Set-up monitor on the fly
    # from psychopy import monitors
    # my_monitor = monitors.Monitor(name=config.get('screen', 'monitor_name'))
    # my_monitor.setSizePix(config.get('screen', 'size'))
    # my_monitor.setWidth(config.get('screen', 'physical_screen_size')[0])
    # my_monitor.setDistance(config.get('screen', 'physical_screen_distance'))
    # my_monitor.saveMon()

    index_number = int(input('What is the pp num? [integer or I will crash]: '))
#    practice = inp_func('Start with practice? [y/n, default y]: ') or 'y'
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
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings_simulate.yml'
    else:
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings.yml'

    # # start practice?
    # if practice == 'y':
    #     sess_prac = LearningSession(scanner=scanner,
    #                                 output_str=output_str,
    #                                 output_dir=output_dir,
    #                                 settings_file=settings_file,
    #                                 start_block=start_block,
    #                                 index_number=index_number,
    #                                 practice=True)
    #     sess_prac.run()

    sess = LearningSession(scanner=scanner,
                           output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=start_block,
                           index_number=index_number)
    sess.run()


if __name__ == '__main__':
    main()