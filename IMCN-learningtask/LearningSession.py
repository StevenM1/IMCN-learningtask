from exptools.core.session import MRISession
from LearningStimulus import LearningStimulus, FixationCross
from LearningTrial import LearningTrial, EndOfBlockTrial
from LearningStimulus import LearningStimulus, FixationCross
from psychopy import visual, data
import datetime
import glob
import pandas as pd
import numpy as np
import os
import pyaudio
import subprocess
import wave
from scipy.io import wavfile
import copy
import cPickle as pkl


class LearningSession(MRISession):

    def __init__(self, subject_initials, index_number, tr, start_block, config):
        super(LearningSession, self).__init__(subject_initials,
                                              index_number,
                                              tr=tr,
                                              simulate_mri_trigger=False,
                                              # NB: DO NOT use this MRI simulation option, but rather another!
                                              mri_trigger_key=config.get('mri', 'mri_trigger_key'))

        self.config = config
        self.start_block = start_block  # allows for starting at a later block than 1
        self.warmup_trs = config.get('mri', 'warmup_trs')

        if tr == 2:
            self.trial_duration = 8 - .5
        elif tr == 3:
            self.trial_duration = 9 - 0.5
        if self.subject_initials == 'pilot':
            self.trial_duration = [8.5, 7.5, 8.5, 7.5]

        self.response_button_signs = [config.get('input', 'response_button_left'),
                                      config.get('input', 'response_button_right')]

        _ = self.create_screen(engine='psychopy',
                               size=config.get('screen', 'size'),
                               full_screen=config.get('screen', 'full_screen'),
                               background_color=config.get('screen', 'background_color'),
                               gamma_scale=config.get('screen', 'gamma_scale'),
                               physical_screen_distance=config.get('screen', 'physical_screen_distance'),
                               physical_screen_size=config.get('screen', 'physical_screen_size'),
                               max_lums=config.get('screen', 'max_lums'),
                               wait_blanking=config.get('screen', 'wait_blanking'),
                               screen_nr=config.get('screen', 'screen_nr'),
                               mouse_visible=config.get('screen', 'mouse_visible'))

        # Try this
        # TODO: think about really including this?
        self.screen.recordFrameIntervals = True

        self.phase_durations = np.array([-0.0001,  # wait for scan pulse
                                         1,
                                         1,
                                         1,
                                         1,
                                         1])

        self.load_design()
        self.prepare_objects()

    def load_design(self):

        self.design = pd.DataFrame({'trial_ID': [1, 2, 3, 4, 5, 6],
                                    'stimulus_set': [0, 1, 2, 0, 1, 2],
                                    'reverse_order': [False, False, False, True, True, True],
                                    'block': [0, 0, 0, 0, 0, 0],
                                    'correct_response': [1, 1, 1, 0, 0, 0],
                                    'p_win': [[.8, .2], [.8, .2], [.8, .2], [.8, .2], [.8, .2], [.8, .2]]})

        # fn = 'sub-' + str(self.subject_initials).zfill(3) + '_tr-' + str(self.index_number) + '_design'
        # design = pd.read_csv(os.path.join('designs', fn + '.csv'), sep='\t', index_col=False)
        #
        # self.design = design
        # self.design = self.design.apply(pd.to_numeric)  # cast all to numeric
#        self.design.stop_trial = pd.to_
#        print(self.design)

    def prepare_objects(self):
        """
        Prepares all visual objects (instruction/feedback texts, stimuli)

        """
        config = self.config

        # Fixation cross
        self.fixation_cross = FixationCross(self.screen,
                                            outer_radius=config.get('fixation_cross', 'outer_radius'),
                                            inner_radius=config.get('fixation_cross', 'inner_radius'),
                                            bg=config.get('fixation_cross', 'bg'))

        # Stimuli
        self.stimuli = []
        all_stim = [config.get('stimulus', 'color_set_1'),
                    config.get('stimulus', 'color_set_2'),
                    config.get('stimulus', 'color_set_3')]
        for stim in all_stim:
            self.stimuli.append(
                LearningStimulus(self.screen,
                                 width=config.get('stimulus', 'width'),
                                 height=config.get('stimulus', 'height'),
                                 colors=stim,
                                 units=config.get('stimulus', 'units'),
                                 x_pos=config.get('stimulus', 'x_pos'),
                                 rect_line_width=config.get('stimulus', 'rect_line_width')))

        # load txts for feedback
        this_file = os.path.dirname(os.path.abspath(__file__))
        self.language = 'en'
        with open(os.path.join(this_file, 'instructions', self.language, 'feedback.txt'), 'rb') as f:
            self.feedback_txt = f.read().split('\n\n\n')

        # Prepare feedback stimuli. Rendering of text is supposedly slow so better to do this once only (not every
        # trial)
        self.feedback_text_objects = [
            # 0 = Too slow, no choice
            visual.TextStim(win=self.screen, text=self.feedback_txt[0], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 1 = Too slow, win
            visual.TextStim(win=self.screen, text=self.feedback_txt[1], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 2 = Too slow, no win
            visual.TextStim(win=self.screen, text=self.feedback_txt[2], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 3 = Too fast, win
            visual.TextStim(win=self.screen, text=self.feedback_txt[3], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 4 = Too fast, no win
            visual.TextStim(win=self.screen, text=self.feedback_txt[4], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 5 = In time, win
            visual.TextStim(win=self.screen, text=self.feedback_txt[5], color='darkgreen',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 6 = In time, no win
            visual.TextStim(win=self.screen, text=self.feedback_txt[6], color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),
        ]

        # Waiting for scanner screen
        self.scanner_wait_screen = visual.TextStim(win=self.screen,
                                                   text='Waiting for scanner...',
                                                   name='scanner_wait_screen',
                                                   units=config.get('text', 'units'), font='Helvetica Neue', pos=(0, 0),
                                                   italic=True,
                                                   height=config.get('text', 'height'), alignHoriz='center')

        # if self.subject_initials == 'DEBUG':
        #     self.stop_timing_circle = visual.Circle(win=self.screen,
        #                                             radius=3, edges=50, lineWidth=1.5, fillColor='red',
        #                                             lineColor='red', units='deg',
        #                                             lineColorSpace='rgb', fillColorSpace='rgb')

    def save_data(self, trial_handler=None, block_n='all'):

        output_fn_dat = self.output_file + '_block-' + str(block_n)
        output_fn_frames = self.output_file + '_block-' + str(block_n)

        if trial_handler is not None:
            trial_handler.saveAsPickle(output_fn_dat)
            trial_handler.saveAsWideText(output_fn_dat + '.csv', )

        if self.screen.recordFrameIntervals:
            # Save frame intervals to file
            self.screen.saveFrameIntervals(fileName=output_fn_frames + '_frameintervals.log', clear=False)

            # import matplotlib.pyplot as plt
            # # Make a nice figure
            # intervals_ms = np.array(self.screen.frameIntervals) * 1000
            # m = np.mean(intervals_ms)
            # sd = np.std(intervals_ms)
            #
            # msg = "Mean=%.1fms, s.d.=%.2f, 99%%CI(frame)=%.2f-%.2f"
            # dist_string = msg % (m, sd, m - 2.58 * sd, m + 2.58 * sd)
            # n_total = len(intervals_ms)
            # n_dropped = sum(intervals_ms > (1.5 * m))
            # msg = "Dropped/Frames = %i/%i = %.3f%%"
            # dropped_string = msg % (n_dropped, n_total, 100 * n_dropped / float(n_total))
            #
            # # plot the frame intervals
            # plt.figure(figsize=[12, 8])
            # plt.subplot(1, 2, 1)
            # plt.plot(intervals_ms, '-')
            # plt.ylabel('t (ms)')
            # plt.xlabel('frame N')
            # plt.title(dropped_string)
            #
            # plt.subplot(1, 2, 2)
            # plt.hist(intervals_ms, 50, normed=0, histtype='stepfilled')
            # plt.xlabel('t (ms)')
            # plt.ylabel('n frames')
            # plt.title(dist_string)
            # plt.savefig(output_fn_frames + '_frameintervals.png')

    def close(self):
        """ Saves stuff and closes """

        self.save_data()
        super(LearningSession, self).close()


    def run(self):
        """ Runs this Stop Signal task"""

        self.block_start_time = 0

        # start emulator TODO REMOVE THIS STUFF!!
        # n_vols = [343+2, 513+2, 343+2, 513+2]
        # trs = [3, 2, 3, 2]
        # n_vols = [31+2, 21+2]
        # trs = [3, 2]
        # from psychopy.hardware.emulator import launchScan

        for block_n in np.unique(self.design.block):
            if block_n < self.start_block:
                continue
            this_block_design = self.design.loc[self.design.block == block_n]

            # scanner_emulator = launchScan(win=self.screen, settings={'TR': trs[block_n-1], 'volumes': n_vols[block_n-1],
            #                                                          'sync': 't'},
            #                               mode='Test')

            if isinstance(self.trial_duration, list):
                trial_duration = self.trial_duration[block_n-1]
            else:
                trial_duration = self.trial_duration

            trial_handler = data.TrialHandler(this_block_design.to_dict('records'),
                                              nReps=1,
                                              method='sequential')

            for block_trial_ID, this_trial_info in enumerate(trial_handler):

                this_trial_parameters = {'stimulus_set': int(this_trial_info['stimulus_set']),
                                         'reverse_order': bool(this_trial_info['reverse_order']),
                                         'block': block_n,
                                         'block_trial_ID': block_trial_ID,
                                         'correct_response': this_trial_info['correct_response'],
                                         'p_win': this_trial_info['p_win']}

                these_phase_durations = self.phase_durations.copy()
#                these_phase_durations[1] = this_trial_info.jitter
                # NB we stop the trial 0.5s before the start of the new trial, to allow sufficient computation time
                # for preparing the next trial. Therefore 8.5s instead of 9s.
                # these_phase_durations[3] = trial_duration - these_phase_durations[1] - these_phase_durations[2]

                this_trial = LearningTrial(ID=int(this_trial_info.trial_ID),
                                           parameters=this_trial_parameters,
                                           phase_durations=these_phase_durations,
                                           session=self,
                                           screen=self.screen)

                # run the prepared trial
                this_trial.run()

                # Record some stuff
                trial_handler.addData('rt', this_trial.response['rt'])
                trial_handler.addData('response', this_trial.response['button'])

                # absolute times since session start
                trial_handler.addData('start_time', this_trial.start_time)
                trial_handler.addData('t_time', this_trial.t_time)
                trial_handler.addData('jitter_time', this_trial.jitter_time)
                trial_handler.addData('stimulus_time', this_trial.stimulus_time)
                trial_handler.addData('selection_time', this_trial.selection_time)
                trial_handler.addData('feedback_time', this_trial.feedback_time)
                trial_handler.addData('iti_time', this_trial.iti_time)

                trial_handler.addData('phase_0_measured', this_trial.t_time - this_trial.start_time)
                trial_handler.addData('phase_1_measured', this_trial.jitter_time - this_trial.t_time)
                trial_handler.addData('phase_2_measured', this_trial.stimulus_time - this_trial.jitter_time)
                trial_handler.addData('phase_3_measured', this_trial.selection_time - this_trial.stimulus_time)
                trial_handler.addData('phase_4_measured', this_trial.feedback_time - this_trial.selection_time)
                trial_handler.addData('phase_5_measured', this_trial.iti_time - this_trial.feedback_time)

                # durations / time since actual start of the block. These are useful to create events-files later for
                #  convolving. Can also grab these from the eventArray though.
                # trial_handler.addData('trial_t_time_block_measured', this_trial.t_time - self.block_start_time)
                # trial_handler.addData('stimulus_onset_time_block_measured', this_trial.jitter_time -
                #                       self.block_start_time)
                # Counter-intuitive, but jitter_time is END of the jitter period = onset of stim

                # # Update staircase if this was a stop trial
                # if is_stop_trial:
                #     if this_trial.response_measured:
                #         # Failed stop: Decrease SSD
                #         self.stairs[this_trial_staircase_id].addData(1)
                #     else:
                #         # Successful stop: Increase SSD
                #         self.stairs[this_trial_staircase_id].addData(0)

                if self.stopped:
                    # out of trial
                    break

            # Save
            self.save_data(trial_handler, block_n)

            if self.stopped:
                # out of block
                break

            # end of block
            this_trial = EndOfBlockTrial(ID=int('999' + str(block_n)),
                                         parameters={},
                                         phase_durations=[0.5, 1000],
                                         session=self,
                                         screen=self.screen)
            this_trial.run()

        self.close()




if __name__ == '__main__':
    from psychopy import core

    # Load config
    from exptools.utils.config import ExpToolsConfig
    config = ExpToolsConfig()

    # Set-up monitor on the fly
    from psychopy import monitors
    my_monitor = monitors.Monitor(name=config.get('screen', 'monitor_name'))
    my_monitor.setSizePix(config.get('screen', 'size'))
    my_monitor.setWidth(config.get('screen', 'physical_screen_size')[0])
    my_monitor.setDistance(config.get('screen', 'physical_screen_distance'))
    my_monitor.saveMon()

    # Set-up session
    sess = LearningSession('DEBUG',
                           1,
                           tr=3,
                           start_block=0,
                           config=config)

    # EMULATOR
    from psychopy.hardware.emulator import launchScan
    scanner_emulator = launchScan(win=sess.screen, settings={'TR': 3, 'volumes': 30000, 'sync': 't'}, mode='Test')

    # run
    sess.run()

    # Load & dump data
    import cPickle as pkl
    from pprint import pprint

    with open(sess.output_file + '_outputDict.pkl', 'r') as f:
        a = pkl.load(f)
    pprint(a)

    core.quit()