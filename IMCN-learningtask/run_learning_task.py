
from LearningSession import *
import datetime
import sys

if sys.version[0] == '2':
    inp_func = raw_input
else:
    inp_func = input

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

    index_number = int(inp_func('What is the pp num? [integer or I will crash]: '))
#    practice = inp_func('Start with practice? [y/n, default y]: ') or 'y'
    start_block = inp_func('Start block? [default 1]: ')
    try:
        start_block = int(start_block)
    except:
        start_block = 1

    if start_block > 1:
        pass
        # ToDo: find last run data to get points from

    scanner = inp_func('Are you in the scanner? [y/n, default n]: ') or 'n'
    while not scanner in ['n', 'y']:
        print('I don''t understand that. Please enter ''y'' or ''n''.')
        scanner = inp_func('Are you in the scanner? [y/n, default n]: ') or 'n'

    simulate = 'n'
    if scanner == 'n':
        simulate = inp_func('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'
        while not simulate in ['n', 'y']:
            print('I don''t understand that. Please enter ''y'' or ''n''.')
            simulate = inp_func('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'

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

    # Force python to quit (so scanner emulator also stops)
    # core.quit()




#
#
# # Set-up session
# sess = StopSignalSession('DEBUG',
#                          1,
#                          run=1,
#                          tr=3,
#                          config=config)
#
# # EMULATOR
# from psychopy.hardware.emulator import launchScan
# scanner_emulator = launchScan(win=sess.screen, settings={'TR': 2, 'volumes': 30000, 'sync': 't'}, mode='Test')
#
# # run
# sess.run()
#
# # Load & dump data
# import cPickle as pkl
# from pprint import pprint
#
# with open(sess.output_file + '_outputDict.pkl', 'r') as f:
#     a = pkl.load(f)
# pprint(a)
#
# core.quit()