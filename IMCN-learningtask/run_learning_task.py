
from LearningSession import *
from psychopy import core

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
    # Load config
    import glob
    import datetime
    from exptools.utils.config import ExpToolsConfig
    config = ExpToolsConfig()

    # Set-up monitor on the fly
    from psychopy import monitors
    my_monitor = monitors.Monitor(name=config.get('screen', 'monitor_name'))
    my_monitor.setSizePix(config.get('screen', 'size'))
    my_monitor.setWidth(config.get('screen', 'physical_screen_size')[0])
    my_monitor.setDistance(config.get('screen', 'physical_screen_distance'))
    my_monitor.saveMon()

    tr = 0
    initials = raw_input('Your initials/subject number: ')
    index_num = int(raw_input('What is the pp num? [integer or I will crash]: '))
    practice = raw_input('Start with practice? [y/n, default y]: ') or 'y'
    start_block = raw_input('Start block? [default 1]: ') or 1
    if start_block > 1:
        pass
        # ToDo: find last run data to get points from

    scanner = raw_input('Are you in the scanner? [y/n, default n]: ') or 'n'
    while not scanner in ['n', 'y']:
        print('I don''t understand that. Please enter ''y'' or ''n''.')
        scanner = raw_input('Are you in the scanner? [y/n, default n]: ') or 'n'

    simulate = 'n'
    if scanner == 'n':
        simulate = raw_input('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'
        while not simulate in ['n', 'y']:
            print('I don''t understand that. Please enter ''y'' or ''n''.')
            simulate = raw_input('Do you want to simulate scan pulses? [y/n, default n]: ') or 'n'

    if scanner == 'y' or simulate == 'y':
        tr = float(raw_input('What is the TR?: ')) or 0

    ### start practice?
    if practice == 'y':
        sess_prac = LearningSession(subject_initials=initials,
                                    index_number=index_num,
                                    tr=tr,
                                    start_block=start_block,
                                    config=config,
                                    practice=True)
        sess_prac.run()

    sess = LearningSession(subject_initials=initials,
                           index_number=index_num,
                           tr=tr,
                           start_block=start_block,
                           config=config,
                           practice=False)

    if simulate == 'y':
        # Run with simulated scanner (useful for behavioral pilots with eye-tracking)
        from psychopy.hardware.emulator import launchScan
        scanner_emulator = launchScan(win=sess.screen, settings={'TR': tr, 'volumes': 30000, 'sync': 't'},
                                      mode='Test')
    sess.run()


if __name__ == '__main__':
    main()

    # Force python to quit (so scanner emulator also stops)
    core.quit()




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