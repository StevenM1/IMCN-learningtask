from exptools2.core.session import Session
from LearningTrial import LearningTrial, EndOfBlockTrial
from LearningStimulus import FixationCross, LearningStimulusSingle, SelectionRectangle
from psychopy import visual
import pandas as pd
import numpy as np
import os
import os.path as op


class LearningSession(Session):

    def __init__(self, index_number, output_str, output_dir, settings_file,
                 start_block, debug=False, scanner=False):
        """ Scanner is a bool """
        super(LearningSession, self).__init__(output_str=output_str,
                                              output_dir=output_dir,
                                              settings_file=settings_file)

        self.index_number = index_number  # participant number
        self.start_block = start_block    # allows for starting at a later block than 1
        self.debug = debug
        self.in_scanner = scanner
        self.design = None
        self.p_wins = None
        self.trials = None
        self.stimuli = None
        self.fixation_cross = None
        self.total_points = 0
        self.total_trials = 0

        self.response_button_signs = [self.settings['input']['response_button_left'],
                                      self.settings['input']['response_button_right']]

        # note that all these durations are default values and can and should be overwritten before each trial
        self.phase_durations = np.array([100,   # phase 0: wait for trigger
                                         1,     # phase 1: fix cross1 (ITI pretrial)
                                         1,     # phase 2: cue
                                         1,     # phase 3: fix cross2
                                         2,     # phase 4: stimulus
                                         0.5,   # phase 5: fix cross3
                                         0.5,   # phase 6: choice highlight
                                         0.5,   # phase 7: fix cross4
                                         0.5,   # phase 8: feedback
                                         1      # phase 9: fix cross5 (ITI posttrial)
                                         ])
        self.phase_names = ['fix_cross_1',
                            'cue',
                            'fix_cross_2',
                            'stimulus',
                            'fix_cross_3',
                            'highlight',
                            'fix_cross_4',
                            'feedback',
                            'fix_cross_5']

    def load_design(self):

        fn = 'sub-' + str(self.index_number).zfill(2) + '_design'

        self.design = pd.read_csv(os.path.join('designs', fn + '.csv'), sep='\t')
        self.p_wins = self.design.p_win_correct.unique()

    def estimate_bonus(self, return_final=False,
                       n_trials_total=None,
                       points_per_win=100,
                       max_reward=5,
                       guess_rate=0.5):
        """
        """

        # if return_final:
        #     # n points if *always* chosen the right answer, based on *all* trials
        #     max_points = self.design.shape[0] * np.mean(self.p_wins) * 100.
        # else:
        #     # expected n points if *always* chosen the right answer, based on the number of trials so far
        #     max_points = self.total_trials * 0.5 * 100.
        #
        # n_moneys = self.total_points * (10. / max_points) - 10 / 2.
        # n_moneys_capped = np.min([np.max([n_moneys, 0]), 5])  # cap at [0, 5]
        #
        # if return_final:
        #     return max_points, n_moneys_capped
        # else:
        #     return n_moneys_capped

        if n_trials_total is None:
            n_trials_total = self.total_trials

        if n_trials_total == 0:
            print('No trials were run, so no bonus')
            return 0, 0

        max_points = n_trials_total * points_per_win
        reward_per_point = max_reward / max_points

        # correct slope for guessing
        reward_per_point = reward_per_point / guess_rate

        # correct intercept for guessing
        guess_points = max_points * guess_rate

        # estimate bonus
        bonus = (self.total_points - guess_points) * reward_per_point  # 0.02

        # cap
        bonus = np.maximum(np.minimum(bonus, max_reward), 0)

        return bonus, max_points

    def prepare_objects(self):
        """
        Prepares all visual objects (instruction/feedback texts, stimuli)
        """

        settings = self.settings

        # Fixation cross
        self.fixation_cross = FixationCross(self.win,
                                            outer_radius=settings['fixation_cross']['outer_radius'],
                                            inner_radius=settings['fixation_cross']['inner_radius'],
                                            bg=settings['fixation_cross']['bg'])
        self.default_fix = self.fixation_cross  # overwrite

        # checkout if stimulus type is interpreted
        # if not settings['stimulus']['type'] in ['agathodaimon']:
        #     raise(IOError('No idea what stimulus type I should draw. '
        #                   'You entered %s' % settings['stimulus']['type']))
        # checkout if colors exist
        # if settings['stimulus']['type'] == 'colors':
        #     import matplotlib.colors as mcolors
        #     for set_n in ['set_1', 'set_2', 'set_3']:
        #         for col in settings['stimulus'][set_n]:
        #             if not col in mcolors.CSS4_COLORS.keys():
        #                 raise(IOError('I dont understand color %s that was '
        #                               'provided to stimulus set %s...' %(col, set_n)))

        # load all possible stimuli
        # all_stim = []
        # for set_n in range(10):
        #     # assume a maximum of 10 sets, bit arbitrary but seems enough
        #     if config.has_option('stimulus', 'set_%d' % set_n):
        #         all_stim.append(config.get('stimulus', 'set_%d' % set_n))
        #
        # if counterbalance:
        #     # counterbalance, based on index_num
        #     ###### WARNING: This is set-up to be specific for my experiment #######
        #     all_stim = self.counterbalance_stimuli(all_stim)

        # let's find all stimuli first from dataframe
        all_stimuli = np.unique(np.hstack([self.design.stim_high.unique(), self.design.stim_low.unique()]))

        self.stimuli = {'left': {}, 'right': {}}
        for stimulus in all_stimuli:
            for i, location in enumerate(['left', 'right']):
                self.stimuli[location][stimulus] = LearningStimulusSingle(
                    win=self.win,
                    stimulus=stimulus,
                    stimulus_type='agathodaimon',
                    width=settings['stimulus']['width'],
                    height=settings['stimulus']['height'],
                    text_height=settings['stimulus']['text_height'],
                    units=settings['stimulus']['units'],
                    x_pos=settings['stimulus']['x_pos'][i],
                    rect_line_width=settings['stimulus']['rect_line_width'])

        self.selection_rectangles = [
            SelectionRectangle(win=self.win,
                               width=settings['stimulus']['width'],
                               height=settings['stimulus']['height'],
                               text_height=settings['stimulus']['text_height'],
                               units=settings['stimulus']['units'],
                               x_pos=settings['stimulus']['x_pos'][0],
                               rect_line_width=settings['stimulus']['rect_line_width']),
            SelectionRectangle(win=self.win,
                               width=settings['stimulus']['width'],
                               height=settings['stimulus']['height'],
                               text_height=settings['stimulus']['text_height'],
                               units=settings['stimulus']['units'],
                               x_pos=settings['stimulus']['x_pos'][1],
                               rect_line_width=settings['stimulus']['rect_line_width'])
        ]

        # # Stimuli
        # self.stimuli = []
        # for stim in all_stim:
        #     self.stimuli.append(
        #         LearningStimulus(self.screen,
        #                          stimulus_type=config.get('stimulus', 'type'),
        #                          width=config.get('stimulus', 'width'),
        #                          height=config.get('stimulus', 'height'),
        #                          set=stim,
        #                          text_height=config.get('stimulus', 'text_height'),
        #                          units=config.get('stimulus', 'units'),
        #                          x_pos=config.get('stimulus', 'x_pos'),
        #                          rect_line_width=config.get('stimulus', 'rect_line_width')))

        # load instruction screen pdfs
        self.instruction_screens = [
            visual.ImageStim(self.win, image='./lib/vanilla.png'),
            visual.ImageStim(self.win, image='./lib/sat.png')
        ]

        # Prepare feedback stimuli. Rendering of text is supposedly slow so better to do this once only (not every
        # trial)
        # 1. Feedback on outcomes
        self.feedback_outcome_objects = [
            visual.TextStim(win=self.win, text='Outcome: 0',  color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][0])),
            visual.TextStim(win=self.win, text='Outcome: +100', color='darkgreen',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][0])),
            visual.TextStim(win=self.win, text='No choice made', color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][0])),
        ]
        # 2. Feedback on earnings
        self.feedback_earnings_objects = [
            visual.TextStim(win=self.win, text='Reward: 0', color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][1])),
            visual.TextStim(win=self.win, text='Reward: +100', color='darkgreen',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][1])),
            visual.TextStim(win=self.win, text='Reward: -100', color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][1])),
        ]
        # 3. Feedback on timing
        self.feedback_timing_objects = [
            visual.TextStim(win=self.win, text='Too slow!', color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][2])),
            visual.TextStim(win=self.win, text='Too fast!', color='darkred',
                            units=self.settings['text']['units'],
                            height=self.settings['text']['height'],
                            pos=(0, self.settings['text']['feedback_y_pos'][2]))]

        # Prepare cue texts. Rendering of text is supposedly slow so better to do this once only (not every
        # trial)
        self.cues = [
            # 0 = SPD
            visual.TextStim(win=self.win, text="SPD",
                            units=settings['text']['units'],
                            height=settings['text']['height']),

            # 1 = ACC
            visual.TextStim(win=self.win, text="ACC",
                            units=settings['text']['units'],
                            height=settings['text']['height']),
        ]

        # Waiting for scanner screen
        self.scanner_wait_screen = visual.TextStim(win=self.win,
                                                   text='Waiting for scanner...',
                                                   name='scanner_wait_screen',
                                                   units=settings['text']['units'],
                                                   height=settings['text']['height'],
                                                   font='Helvetica Neue', pos=(0, 0),
                                                   italic=True,
                                                   alignHoriz='center')

        if self.debug:
             pos = -settings['window']['size'][0]/3, settings['window']['size'][1]/3
             self.debug_txt = visual.TextStim(win=self.win,
                                              alignVert='top',
                                              text='debug mode\n',
                                              name='debug_txt',
                                              units='pix',  # config.get('text', 'units'),
                                              font='Helvetica Neue',
                                              pos=pos,
                                              height=14,  # config.get('text', 'height'),
                                              alignHoriz='center')

    def save_data(self, block_nr=None):

        global_log = pd.DataFrame(self.global_log).set_index('trial_nr').copy()
        global_log['onset_abs'] = global_log['onset'] + self.exp_start

        # Only non-responses have a duration
        nonresp_idx = ~global_log.event_type.isin(['response', 'trigger', 'pulse', 'non_response_keypress'])
        last_phase_onset = global_log.loc[nonresp_idx, 'onset'].iloc[-1]

        if block_nr is None:
            dur_last_phase = self.exp_stop - last_phase_onset
        else:
            dur_last_phase = self.clock.getTime() - last_phase_onset
        durations = np.append(global_log.loc[nonresp_idx, 'onset'].diff().values[1:], dur_last_phase)
        global_log.loc[nonresp_idx, 'duration'] = durations

        # Same for nr frames
        nr_frames = np.append(global_log.loc[nonresp_idx, 'nr_frames'].values[1:], self.nr_frames)
        global_log.loc[nonresp_idx, 'nr_frames'] = nr_frames.astype(int)

        # Round for readability and save to disk
        global_log = global_log.round({'onset': 5, 'onset_abs': 5, 'duration': 5})

        if block_nr is None:
            f_out = op.join(self.output_dir, self.output_str + '_events.tsv')
        else:
            f_out = op.join(self.output_dir, self.output_str + '_block-' + str(block_nr) + '_events.tsv')
        global_log.to_csv(f_out, sep='\t', index=True)

        # Save frame intervals to file
        self.win.saveFrameIntervals(fileName=f_out.replace('_events.tsv', '_frameintervals.log'), clear=False)

    def close(self):
        """ 'Closes' experiment. Should always be called, even when
        experiment is quit manually (saves onsets to file). """

        if self.closed:  # already closed!
            return None

        # self.win.saveMovieFrames(fileName='frames/DEMO2.png')

        self.win.callOnFlip(self._set_exp_stop)
        self.win.flip()
        self.win.recordFrameIntervals = False

        print(f"\nDuration experiment: {self.exp_stop:.3f}\n")

        if not op.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # points = self.total_points
        bonus, max_points = self.estimate_bonus()
        if not max_points == 0:
            perc = self.total_points/max_points*100
        else:
            perc = 0

        print('Total points: %d (maximum: %d; so %.3f pct), total bonus earned: %.3f' % (self.total_points,
                                                                                         max_points,
                                                                                         perc,
                                                                                         bonus))

        self.save_data()

        # Create figure with frametimes (to check for dropped frames)
        # fig, ax = plt.subplots(figsize=(15, 5))
        # ax.plot(self.win.frameIntervals)
        # ax.axhline(1. / self.actual_framerate, c='r')
        # ax.axhline(1. / self.actual_framerate + 1. / self.actual_framerate, c='r', ls='--')
        # ax.set(xlim=(0, len(self.win.frameIntervals) + 1), xlabel='Frame nr', ylabel='Interval (sec.)',
        #        ylim=(-0.1, 0.5))
        # fig.savefig(op.join(self.output_dir, self.output_str + '_frames.png'))

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        self.win.close()
        self.closed = True

    def create_trials(self, block_nr):
        this_block_design = self.design.loc[self.design.block == block_nr]
        self.trials = []

        for i, (index, row) in enumerate(this_block_design.iterrows()):
            if row['iti_posttrial'] >= 6:
                n_trs = 5
            else:
                n_trs = 3

            this_trial_parameters = {'stimulus_symbol_left': row['stim_left'],
                                     'stimulus_symbol_right': row['stim_right'],
                                     'correct_response': row['correct_stim_lr'],
                                     'block_nr': block_nr,
                                     'p_win_left': row['p_win_left'],
                                     'p_win_right': row['p_win_right'],
                                     'cue': row['cue_txt'],
                                     'n_trs': n_trs}

            phase_durations = [row['fix_cross_1'],
                               row['cue'],
                               row['fix_cross_2'],
                               row['stimulus'],
                               row['fix_cross_3'],
                               row['highlight'],
                               row['fix_cross_4'],
                               row['feedback'],
                               100]   # show fixation cross 5 until scanner sync!

            self.trials.append(LearningTrial(trial_nr=int(index),
                                             parameters=this_trial_parameters,
                                             phase_durations=phase_durations,
                                             phase_names=self.phase_names,
                                             session=self))

            if self.debug:
                if i >= 10:
                    break

    def run(self, quit_on_exit=True):
        """ Runs this Instrumental Learning task"""

        self.load_design()
        self.prepare_objects()

        # remove blocks before start_block
        all_blocks = np.unique(self.design.block)
        all_blocks = all_blocks[self.start_block-1:]  # assuming start_block is 1-coded

        for block_nr in all_blocks:
            self.create_trials(block_nr)

            if self.exp_start is None:
                self.start_experiment()

            self.display_text('Waiting for scanner', keys=self.mri_trigger)
            self.timer.reset()

            # loop over trials
            for trial in self.trials:
                trial.run()
                self.total_trials += 1
                self.total_points += trial.points_earned

            # save data
            self.save_data(block_nr=block_nr)

            # show end of block screen
            if block_nr == 3:
                tr = EndOfBlockTrial(trial_nr=self.total_trials + 10000, parameters={},
                                     phase_durations=[1000], exp_end=True,
                                     phase_names=['show_text'], session=self)
                tr.run()
            else:
                tr = EndOfBlockTrial(trial_nr=self.total_trials + 10000, parameters={},
                                     phase_durations=[1000],
                                     phase_names=['show_text'], session=self)
                tr.run()

        self.close()
        if quit_on_exit:
            self.quit()


if __name__ == '__main__':

    import datetime
    index_number = 2
    start_block = 1
    scanner = True
    simulate = 'y'
    debug = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{index_number}_task-learning_datetime-{timestamp}'
    output_dir = './data'
    if simulate == 'y':
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings_simulate.yml'
    else:
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings.yml'

    sess = LearningSession(scanner=scanner,
                           output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=start_block,
                           debug=debug,
                           index_number=index_number)
    sess.run()

# if __name__ == '__main__':
#     from psychopy import core
#
#     # Load config
#     from exptools.utils.config import ExpToolsConfig
#     config = ExpToolsConfig()
#
#     # Set-up monitor on the fly
#     from psychopy import monitors
#     my_monitor = monitors.Monitor(name=config.get('screen', 'monitor_name'))
#     my_monitor.setSizePix(config.get('screen', 'size'))
#     my_monitor.setWidth(config.get('screen', 'physical_screen_size')[0])
#     my_monitor.setDistance(config.get('screen', 'physical_screen_distance'))
#     my_monitor.saveMon()
#
#     # sub_id = 'PRACTICE'
#     # Set-up session
#     sess = LearningSession('1',
#                            1,
#                            tr=0,
#                            start_block=0,
#                            config=config,
#                            debug=True,
#                            practice=True)
#
#     # EMULATOR
#     # from psychopy.hardware.emulator import launchScan
#     # scanner_emulator = launchScan(win=sess.screen, settings={'TR': 0.5, 'volumes': 30000, 'sync': 't'}, mode='Test')
#
#     # run
#     sess.run()
#
#     # Load & dump data
#     import cPickle as pkl
#     from pprint import pprint
#
#     with open(sess.output_file + '_outputDict.pkl', 'r') as f:
#         a = pkl.load(f)
#     pprint(a)
#
#     core.quit()






#
        #     def counterbalancounterbalancece_stimuli(self, all_sets):
#         """
#         For counterbalancing: determine 'order' of the provided stimulus set. That is, we want to make sure that a
#         specific character has varying probabilities of 'winning' over participants. E.g., assuming your stimuli are
#         ABCDEFGHIJKL, and you want ABCDEF and GHIJKL never to intermix, you may want this:
#
# pp  stim_1    stim_2    stim_3    stim_4    stim_5    stim_6    stim_7    stim_8    stim_9    stim_10    stim_11    stim_12
# --  ----  --------  --------  --------  --------  --------  --------  --------  --------  --------  ---------  ---------  ---------
#  0     1  A         B         C         D         E         F         G         H         I         J          K          L
#  1     2  B         A         D         C         F         E         H         G         J         I          L          K
#  2     3  C         D         E         F         A         B         I         J         K         L          G          H
#  3     4  D         C         F         E         B         A         J         I         L         K          H          G
#  4     5  E         F         A         B         C         D         K         L         G         H          I          J
#  5     6  F         E         B         A         D         C         L         K         H         G          J          I
#  6     7  G         H         I         J         K         L         A         B         C         D          E          F
#  7     8  H         G         J         I         L         K         B         A         D         C          F          E
#  8     9  I         J         K         L         G         H         C         D         E         F          A          B
#  9    10  J         I         L         K         H         G         D         C         F         E          B          A
# 10    11  K         L         G         H         I         J         E         F         A         B          C          D
# 11    12  L         K         H         G         J         I         F         E         B         A          D          C
#         """
#         import itertools
#         from copy import deepcopy
#
#         n_shifts = [0, 1, 2]  # assuming 6 stimuli, but you could do more, or less...
#         rev_inner = [False, True]
#         switch_sets = [False, True]
#
#         cb_df = pd.DataFrame(list(itertools.product(switch_sets, n_shifts, rev_inner)),
#                              columns=['switch_sets', 'n_shifts', 'rev_inner'])
#         cb_df['pp'] = np.arange(1, cb_df.shape[0] + 1)
#         for set_n in range(1, 13):
#             cb_df['stim_%d' % set_n] = None
#
#         for pp in cb_df['pp']:
#             idx = cb_df.pp == pp
#             switch_sets = cb_df.loc[idx, 'switch_sets'].iloc[0]
#             reverse_inner = cb_df.loc[idx, 'rev_inner'].iloc[0]
#             n_shifts = cb_df.loc[idx, 'n_shifts'].iloc[0]
#
#             if switch_sets:
#                 sets = deepcopy([all_sets[3:], all_sets[:3]])
#             else:
#                 sets = deepcopy([all_sets[:3], all_sets[3:]])
#
#             sets_allocated = 0
#             for set_n, set_ in enumerate(sets):
#                 for i in range(n_shifts):
#                     set_.insert(len(set_), set_.pop(0))  # move first item to last place
#
#                 if reverse_inner:
#                     set_ = [x[::-1] for x in set_]  # reverse inner order
#
#                 # print('pp %d, %d, %s' % (pp, set_n, set_))
#                 #### NB: you could just use set_ as a final result; the placing in the dataframe and then reverting
#                 # back to a nested list is definitely not necessary but may help clarify what's going on here...
#                 for to_allocate in [0, 1, 2]:
#                     for to_allocate_i in [0, 1]:
#                         cb_df.loc[idx, 'stim_%d' % (sets_allocated + 1)] = set_[to_allocate][to_allocate_i]
#                         sets_allocated += 1
#
#         pp_zero_based = self.index_number - 1
#         row_iloc = int(pp_zero_based - np.floor(pp_zero_based / 12) * 12)
#         colnames = cb_df.columns
#         stim_list = cb_df.iloc[row_iloc][[x for x in colnames if 'stim' in x]].values.tolist()
#         stim_nested_list = [[stim_list[0 + y * 2], stim_list[1 + y * 2]] for y in range(6)]
#         print('Stimuli/set order for this pp: %s' % stim_nested_list)
#         return stim_nested_list



#
#     def update_instruction_screen(self, task='SAT',
#                                   block=0,
#                                   show_upcoming_stimuli=False,
#                                   experiment_start=False,
#                                   end_of_block=False,
#                                   end_of_session=False):
#         """
#         Updates instruction screen based on upcoming block
#         :param task: ['SAT', 'vanilla']
#         :param block: [0, 1, 2]
#         :param experiment_start: bool  (is this the start of the experiment? If true, shows welcome screen)
#         """
#
#         if experiment_start:
#             # Session starts
#             if self.practice:
#                 txt = 'Welcome to this practice session!'
#             else:
#                 txt = 'Welcome to this experiment!'
#             self.current_instruction_screen = [
#                 visual.TextStim(win=self.win,
#                                 text=txt,
#                                 height=self.settings['text']['height'],
#                                 units=self.settings['text']['units'],
#                                 pos=(0, 2)),
#                 visual.TextStim(win=self.win,
#                                 text='Press <space bar> to continue',
#                                 height=self.settings['text']['height'],
#                                 units=self.settings['text']['units'],
#                                 pos=(0, -2))
#             ]
#         elif end_of_session:
#             # Session ends
#             if self.practice:
#                 txt = 'This is the end of the practice session\n\nPress <space bar> to continue to the real experiment'
#             else:
#                 txt = 'This is the end of experiment\n\nPlease inform the experiment leader now'
#             self.current_instruction_screen = [
#                 visual.TextStim(self.win,
#                                 pos=(0, 0),
#                                 height=self.settings['text']['height'], #'.get('text', 'height'),
#                                 units=self.settings['text']['units'],
#                                 text=txt,
#                                 wrapWidth=80
#                                 )]
#
#         elif block in [1, 4] and not show_upcoming_stimuli:
#             # Announce vanilla / SAT task
#             if task == 'SAT':
#                 if self.practice:
#                     self.current_instruction_screen = [
#                         visual.TextStim(self.win,
#                                         text='You will next practice the task with speed/accuracy cues. We will '
#                                              'again first show you what all of the stimuli in the upcoming block look '
#                                              'like.',
#                                         pos=(0, 0),
#                                         height=self.settings['text']['height'],
#                                         units=self.settings['text']['units'],
#                                         wrapWidth=self.settings['text']['wrap_width']
#                                         ),
#                         visual.TextStim(self.win,
#                                         text='Press <space bar> to continue',
#                                         pos=(0, -8),
#                                         italic=True,
#                                         height=self.settings['text']['height'],
#                                         units=self.settings['text']['units'],
#                                         wrapWidth=self.settings['text']['wrap_width']
#                                         )
#                     ]
#                 else:
#                     # prep screen here
#                     self.current_instruction_screen = [
#                         self.instruction_screens[1]
#                     ]
#             elif task == 'vanilla':
#                 if self.practice:
#                     self.current_instruction_screen = [
#                         visual.TextStim(self.win,
#                                         text='We will first introduce the task without any speed/accuracy cues.\n\n'
#                                              'To start with, we will show you what all of the stimuli in the upcoming '
#                                              'block look like',
#                                         pos=(0, 0),
#                                         height=self.settings['text']['height'],
#                                         units=self.settings['text']['units'],
#                                         wrapWidth=self.settings['text']['wrap_width']
#                                         ),
#                         visual.TextStim(self.win,
#                                         text='Press <space bar> to continue',
#                                         pos=(0, -8), italic=True,
#                                         height=self.settings['text']['height'],
#                                         units=self.settings['text']['units'],
#                                         wrapWidth=self.settings['text']['wrap_width']
#                                         )
#                     ]
#                 else:
#                     self.current_instruction_screen = [
#                         self.instruction_screens[0]
#                     ]
#
#         elif show_upcoming_stimuli:
#             # show all upcoming stimuli
#             idx = (self.design.block == block)
#             all_upcoming_pairs = self.design.loc[idx].groupby(['stimulus_set'])[['stim_left', 'stim_right']].last().reset_index()[
#                 ['stim_left', 'stim_right']].values.tolist()
#             # all_upcoming_pairs = [self.design.loc[idx, 'stim_left'].unique(), self.design.loc[idx, 'stim_right'].unique()]
#             y_positions = [4, 0, -4]
#
#             if self.practice:
#                 self.current_instruction_screen = [
#                     visual.TextStim(self.win,
#                                     text='The figures below will be your choice options in the next block. Have a '
#                                          'look at them, and try to remember what they look like',
#                                     pos=(0, 8),
#                                     height=self.settings['text']['height'],
#                                     units=self.settings['text']['units'],
#                                     wrapWidth=self.settings['text']['wrap_width']
#                                     )
#                 ]
#             else:
#                 self.current_instruction_screen = [
#                     visual.TextStim(self.win,
#                                     text='In the upcoming new block, you will see new choice pairs, illustrated '
#                                          'below. As '
#                                          'before, you will need to learn which choice options are most valuable. '
#                                          'Have a look at them, and try to remember what they look like',
#                                     pos=(0, 9),
#                                     height=self.settings['text']['height'],
#                                     units=self.settings['text']['units'],
#                                     wrapWidth=self.settings['text']['wrap_width']
#                                     )
#                 ]
#
#             for i, pair in enumerate(all_upcoming_pairs):
#                 self.current_instruction_screen.append(
#                     visual.TextStim(self.win,
#                                     pos=(self.settings['stimulus', 'x_pos'][0], y_positions[i]),
#                                     text=pair[0],
#                                     height=self.settings['stimulus', 'text_height'],
#                                     units=self.settings['text', 'units'])#,
# #                                    font='Agathodaimon',
# #                                    fontFiles=['./lib/AGATHODA.TTF'])
#                 )
#                 self.current_instruction_screen.append(
#                     visual.TextStim(self.win,
#                                     pos=(self.settings['stimulus', 'x_pos'][1], y_positions[i]),
#                                     text=pair[1],
#                                     height=self.settings['stimulus', 'text_height'],
#                                     units=self.settings['text', 'units'])#,
#                                    # font='Agathodaimon', fontFiles=['./lib/AGATHODA.TTF'])
#                 )
#
#             # Boxes!
#             for i, pair in enumerate(all_upcoming_pairs):
#                 self.current_instruction_screen.append(
#                     visual.Rect(self.win, width=14, height=3.5, pos=(0, y_positions[i]), units='deg')
#                 )
#             if self.practice:
#                 self.current_instruction_screen.append(
#                     visual.TextStim(self.win,
#                                     text='The pairs above will also be presented together, as indicated by the '
#                                          'rectangles.\nNext, we will show an example trial, with step-by-step '
#                                          'explanations of what to do',
#                                     pos=(0, -9),
#                                     height=self.settings['text']['height'],
#                                     units=self.settings['text']['units'],
#                                     wrapWidth=self.settings['text']['wrap_width']))
#                 self.current_instruction_screen.append(
#                     visual.TextStim(self.win,
#                                     text='To start the practice trial, press <space bar>!',
#                                     pos=(0, -13), italic=True,
#                                     height=self.settings['text']['height'],
#                                     units=self.settings['text']['units'],
#                                     wrapWidth=self.settings['text']['wrap_width']))
#             else:
#                 self.current_instruction_screen.append(
#                     visual.TextStim(self.win,
#                                     text='To start the task, press <space bar>!',
#                                     pos=(0, -13), italic=True,
#                                     height=self.settings['text']['height'],
#                                     units=self.settings['text']['units'],
#                                     wrapWidth=self.settings['text']['wrap_width']))
#
#         elif end_of_block:
#             # guestimate the amount of money to be earned
#             estimated_moneys = self.estimate_bonus()
#             end_str = '!' if estimated_moneys > 0 else ''
#             break_str = 'You can take a short break now. ' if not self.practice else ''
#
#             self.current_instruction_screen = [
#                 visual.TextStim(self.win, pos=(0, 4),
#                                 text='End of block. So far, you earned %d points%s' % (
#                                 self.total_points, end_str),
#                                 height=self.settings['text']['height'],
#                                 units=self.settings['text']['units'],
#                                 wrapWidth=self.settings['text']['wrap_width']),
#                 visual.TextStim(self.win, pos=(0, 0),
#                                 height=self.settings['text']['height'],
#                                 units=self.settings['text']['units'],
#                                 wrapWidth=self.settings['text']['wrap_width'],
#                                 text='Based on your performance so far, it looks like you will receive a bonus of approximately %.2f euro%s' % (estimated_moneys, end_str)),
#                 visual.TextStim(self.win, pos=(0, -4),
#                                 text='%sPress <space bar> to continue.' % break_str,
#                                 height=self.settings['text']['height'],
#                                 units=self.settings['text']['units'],
#                                 wrapWidth=self.settings['text']['wrap_width'])
#             ]


#                # trial_handler.addData('total_points_earned', self.total_points)

            # trial_handler = data.TrialHandler(this_block_design.to_dict('records'),
            #                                   nReps=1,
            #                                   method='sequential')

            # Loop over trials
            # for block_trial_ID, this_trial_info in enumerate(trial_handler):

                # show instruction screen (do this in inner loop so we can peek into the next cue)
                # if block_n in [1, 4] and block_trial_ID == 0:
                #     if this_trial_info['cue'] in ['SPD', 'ACC']:
                #         self.update_instruction_screen(block=block_n, task='SAT', experiment_start=False)
                #     else:
                #         self.update_instruction_screen(block=block_n, task='vanilla', experiment_start=False)
                #     _ = InstructionTrial(trial_nr=self.instruction_trial_n,
                #                          parameters={},
                #                          phase_durations=[0.5, 1000],
                #                          session=self).run()
                #     self.instruction_trial_n -= 1

                    # also, show upcoming stimuli
                    # self.update_instruction_screen(block=block_n, show_upcoming_stimuli=True)
                    # _ = InstructionTrial(trial_nr=self.instruction_trial_n,
                    #                      parameters={},
                    #                      phase_durations=[0.5, 1000],
                    #                      session=self).run()
                    # self.instruction_trial_n -= 1

                # Actual trial
                # this_trial_parameters = {'stim_left': this_trial_info['stim_left'],
                #                          'stim_right': this_trial_info['stim_right'],
                #                          'correct_response': this_trial_info['correct_stim_lr'],
                #                          'block': block_n,
                #                          'block_trial_ID': block_trial_ID,
                #                          'p_win_left': this_trial_info['p_win_left'],
                #                          'p_win_right': this_trial_info['p_win_right'],
                #                          'cue': this_trial_info['cue']}

                # these_phase_durations = self.phase_durations.copy()
                # for phase_n in np.arange(8):
                #     if 'phase_' + str(phase_n) in this_trial_info.keys():
                #         these_phase_durations[phase_n] = this_trial_info['phase_' + str(phase_n)]

                # NB we stop the trial 0.5s before the start of the new trial, to allow sufficient computation time
                # for preparing the next trial. (but never below 0.1s)
                # these_phase_durations[-1] = np.max([0.1, these_phase_durations[-1]-0.5])
                #
                # this_trial = self.run_trial(trial_nr=int(this_trial_info.trial_ID),
                #                             # annotate=this_trial_info.annotate,
                #                             parameters=this_trial_parameters,
                #                             phase_durations=self.phase_durations, #these_phase_durations,
                #                             phase_names=self.phase_names,
                #                             session=self)
                #
                # # run the prepared trial
                # this_trial.run()
                #
                # # Record some stuff
                # trial_handler.addData('rt', this_trial.response['rt'])
                # trial_handler.addData('response', this_trial.response['button'])
                # trial_handler.addData('deadline', this_trial.deadline)
                #
                # trial_handler.addData('points_earned', this_trial.points_earned)
                # trial_handler.addData('outcome', this_trial.won)
                # self.total_points += this_trial.points_earned
                # self.total_trials += 1
                # trial_handler.addData('total_points_earned', self.total_points)

                # absolute times since session start
                # for time_name in ['start_time', 't_time', 'jitter_time_1', 'cue_time', 'jitter_time_2',
                #                   'stimulus_time', 'selection_time', 'feedback_time', 'iti_time']:
                #     trial_handler.addData(time_name, getattr(this_trial, time_name))

                # durations / time since actual start of the block. These are useful to create events-files later for
                #  convolving. Can also grab these from the eventArray though.
                # trial_handler.addData('trial_t_time_block_measured', this_trial.t_time - self.block_start_time)
                # trial_handler.addData('stimulus_onset_time_block_measured', this_trial.jitter_time -
                #                       self.block_start_time)
                # Counter-intuitive, but jitter_time is END of the jitter period = onset of stim

                # if self.restart_block:
                #     current_block_id -= 1
                #     break

                # check for last trial?
                # if this_trial_info.trial_ID == self.design.trial_ID.max():
                #     self.stopped = True

            # Save
            # self.save_data(trial_handler, block_n)

            # if not self.restart_block:
            #     # end of block, show expected score
            #     self.update_instruction_screen(end_of_block=True)
            #     _ = InstructionTrial(trial_nr=self.instruction_trial_n,
            #                          parameters={},
            #                          phase_durations=[0.5, 1000],
            #                          session=self).run()
            #     self.instruction_trial_n -= 1
            # else:
            #     self.restart_block = False

        # # End of experiment
        # self.update_instruction_screen(end_of_session=True)
        # _ = InstructionTrial(trial_nr=self.instruction_trial_n,
        #                      parameters={},
        #                      phase_durations=[0.5, 1000],
        #                      session=self).run()
        # self.instruction_trial_n -= 1