from exptools.core.session import MRISession
from LearningTrial import LearningTrial, InstructionTrial, PracticeTrial #, EndOfExperimentTrial, EndOfBlockTrial
from LearningStimulus import LearningStimulus, FixationCross, LearningStimulusSingle
from psychopy import visual, data
import datetime
import glob
import pandas as pd
import numpy as np
import os
import copy
import sys
if int(sys.version[0]) == 3:
    import pickle as pkl
else:
    import cPickle as pkl


class LearningSession(MRISession):

    def __init__(self, subject_initials, index_number, tr, start_block, config, debug=False, practice=False):
        super(LearningSession, self).__init__(subject_initials,
                                              index_number,
                                              tr=tr,
                                              simulate_mri_trigger=False,
                                              # NB: DO NOT use this MRI simulation option, but rather another!
                                              mri_trigger_key=config.get('mri', 'mri_trigger_key'))

        self.config = config
        self.start_block = start_block  # allows for starting at a later block than 1
        self.warmup_trs = config.get('mri', 'warmup_trs')
        self.restart_block = False
        self.debug = debug
        self.practice = practice

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

        # TODO: think about really including this?
        self.screen.recordFrameIntervals = True

        # negative durations below are jittered and will be determined per trial (except phase 0)
        # note that all these durations are default values and can and will be overwritten before each trial
        self.phase_durations = np.array([-0.0001,  # phase 0: wait for scan pulse
                                         -0.1,  # phase 1: fix cross
                                         1,     # phase 2: cue
                                         -0.1,  # phase 3: fix cross
                                         2,     # phase 4: stimulus
                                         0.5,   # phase 5: choice highlight
                                         0.5,   # phase 6: feedback
                                         -.01   # phase 7: ITI
                                         ])
        self.total_points = 0
        self.total_trials = 0
        self.instruction_trial_n = -1
        self.load_design()
        self.prepare_objects(counterbalance=True)

    def load_design(self):

        if self.practice:
            fn = 'sub-PRACTICE_design'
        elif self.debug:
            fn = 'sub-DEBUG_design'
        else:
            fn = 'sub-' + str(self.index_number).zfill(2) + '_design'

        self.design = pd.read_csv(os.path.join('designs', fn + '.csv'), sep='\t', index_col=False)
        self.p_wins = self.design.p_win_correct.unique()

        # For practice sessions, we want to annotate some stuff but only in the first three trials per block (?)
        self.design['annotate'] = False
        if self.practice:
            for block in [1, 4]:
                idx = self.design.block == block
                idx = (np.cumsum(idx) > 0) & (np.cumsum(idx) < 2)  # run only the first trial annotated
                self.design.loc[idx, 'annotate'] = True

    def estimate_bonus(self, return_final=False):
        """
        simple linear combination
        y = a*x + b
        a = 10/max_points
        b = -10/2
        """

        if return_final:
            # n points if *always* chosen the right answer, based on *all* trials
            max_points = self.design.shape[0] * np.mean(self.p_wins) * 100.
        else:
            # expected n points if *always* chosen the right answer, based on the number of trials so far
            max_points = self.total_trials * np.mean(self.p_wins) * 100.
        n_moneys = self.total_points * (10. / max_points) - 10 / 2.
        n_moneys_capped = np.min([np.max([n_moneys, 0]), 5])  # cap at [0, 5]

        if return_final:
            return max_points, n_moneys_capped
        else:
            return n_moneys_capped

    def update_instruction_screen(self, task='SAT',
                                  block=0,
                                  show_upcoming_stimuli=False,
                                  experiment_start=False,
                                  end_of_block=False,
                                  end_of_session=False):
        """
        Updates instruction screen based on upcoming block
        :param task: ['SAT', 'vanilla']
        :param block: [0, 1, 2]
        :param experiment_start: bool  (is this the start of the experiment? If true, shows welcome screen)
        """

        if experiment_start:
            # Session starts
            if self.practice:
                txt = 'Welcome to this practice session!'
            else:
                txt = 'Welcome to this experiment!'
            self.current_instruction_screen = [
                visual.TextStim(win=self.screen, text=txt,
                                height=self.config.get('text', 'height'),
                                units=self.config.get('text', 'units'),
                                pos=(0, 2)),
                visual.TextStim(win=self.screen, text='Press <space bar> to continue',
                                height=self.config.get('text', 'height'),
                                units=self.config.get('text', 'units'),
                                pos=(0, -2))
            ]
        elif end_of_session:
            # Session ends
            if self.practice:
                txt = 'This is the end of the practice session\n\nPress <space bar> to continue to the real experiment'
            else:
                txt = 'This is the end of experiment\n\nPlease inform the experiment leader now'
            self.current_instruction_screen = [
                visual.TextStim(self.screen,
                                pos=(0, 0),
                                units=self.config.get('text', 'units'),
                                height=self.config.get('text', 'height'),
                                text=txt,
                                wrapWidth=80
                                )]

        elif block in [1, 4] and not show_upcoming_stimuli:
            # Announce vanilla / SAT task
            if task == 'SAT':
                if self.practice:
                    self.current_instruction_screen = [
                        visual.TextStim(self.screen,
                                        text='You will next practice the task with speed/accuracy cues. We will '
                                             'again first show you what all of the stimuli in the upcoming block look '
                                             'like.',
                                        height=self.config.get('text', 'height'),
                                        units=self.config.get('text', 'units'),
                                        pos=(0, 0),
                                        wrapWidth=self.config.get('text', 'wrap_width')
                                        ),
                        visual.TextStim(self.screen,
                                        text='Press <space bar> to continue',
                                        pos=(0, -8), italic=True,
                                        units=self.config.get('text', 'units'),
                                        height=self.config.get('text', 'height'),
                                        wrapWidth=self.config.get('text', 'wrap_width')
                                        )
                    ]
                else:
                    # prep screen here
                    self.current_instruction_screen = [
                        self.instruction_screens[1]
                    ]
            elif task == 'vanilla':
                if self.practice:
                    self.current_instruction_screen = [
                        visual.TextStim(self.screen,
                                        text='We will first introduce the task without any speed/accuracy cues.\n\n'
                                             'To start with, we will show you what all of the stimuli in the upcoming '
                                             'block look like',
                                        height=self.config.get('text', 'height'),
                                        units=self.config.get('text', 'units'),
                                        pos=(0, 0),
                                        wrapWidth=self.config.get('text', 'wrap_width')
                                        ),
                        visual.TextStim(self.screen,
                                        text='Press <space bar> to continue',
                                        pos=(0, -8), italic=True,
                                        units=self.config.get('text', 'units'),
                                        height=self.config.get('text', 'height'),
                                        wrapWidth=self.config.get('text', 'wrap_width')
                                        )
                    ]
                else:
                    self.current_instruction_screen = [
                        self.instruction_screens[0]
                    ]

        elif show_upcoming_stimuli:
            # show all upcoming stimuli
            idx = (self.design.block == block)
            all_upcoming_pairs = self.design.loc[idx].groupby(['stimulus_set'])[['stim_left', 'stim_right']].last().reset_index()[
                ['stim_left', 'stim_right']].values.tolist()
            # all_upcoming_pairs = [self.design.loc[idx, 'stim_left'].unique(), self.design.loc[idx, 'stim_right'].unique()]
            y_positions = [4, 0, -4]

            if self.practice:
                self.current_instruction_screen = [
                    visual.TextStim(self.screen,
                                    text='The figures below will be your choice options in the next block. Have a '
                                         'look at them, and try to remember what they look like',
                                    pos=(0, 8),
                                    units=self.config.get('text', 'units'),
                                    height=self.config.get('text', 'height'),
                                    wrapWidth=self.config.get('text', 'wrap_width')
                                    )
                ]
            else:
                self.current_instruction_screen = [
                    visual.TextStim(self.screen,
                                    text='In the upcoming new block, you will see new choice pairs, illustrated '
                                         'below. As '
                                         'before, you will need to learn which choice options are most valuable. '
                                         'Have a look at them, and try to remember what they look like',
                                    pos=(0, 9),
                                    units=self.config.get('text', 'units'),
                                    height=self.config.get('text', 'height'),
                                    wrapWidth=self.config.get('text', 'wrap_width')
                                    )
                ]

            for i, pair in enumerate(all_upcoming_pairs):
                self.current_instruction_screen.append(
                    visual.TextStim(self.screen, pos=(self.config.get('stimulus', 'x_pos')[0], y_positions[i]),
                                    text=pair[0], height=self.config.get('stimulus', 'text_height'),
                                    units=self.config.get('text', 'units'),
                                    font='Agathodaimon', fontFiles=['./lib/AGATHODA.TTF'])
                )
                self.current_instruction_screen.append(
                    visual.TextStim(self.screen, pos=(self.config.get('stimulus', 'x_pos')[1], y_positions[i]),
                                    text=pair[1], height=self.config.get('stimulus', 'text_height'),
                                    units=self.config.get('text', 'units'),
                                    font='Agathodaimon', fontFiles=['./lib/AGATHODA.TTF'])
                )

            # Boxes!
            for i, pair in enumerate(all_upcoming_pairs):
                self.current_instruction_screen.append(
                    visual.Rect(self.screen, width=14, height=3.5, pos=(0, y_positions[i]), units='deg')
                )
            if self.practice:
                self.current_instruction_screen.append(
                    visual.TextStim(self.screen,
                                    text='The pairs above will also be presented together, as indicated by the '
                                         'rectangles.\nNext, we will show an example trial, with step-by-step '
                                         'explanations of what to do',
                                    pos=(0, -9),
                                    units=self.config.get('text', 'units'),
                                    height=self.config.get('text', 'height'),
                                    wrapWidth=self.config.get('text', 'wrap_width')
                                    ))
                self.current_instruction_screen.append(
                    visual.TextStim(self.screen,
                                    text='To start the practice trial, press <space bar>!',
                                    pos=(0, -13), italic=True,
                                    units=self.config.get('text', 'units'),
                                    height=self.config.get('text', 'height'),
                                    wrapWidth=self.config.get('text', 'wrap_width')
                                    )
                )
            else:
                self.current_instruction_screen.append(
                    visual.TextStim(self.screen,
                                    text='To start the task, press <space bar>!',
                                    pos=(0, -13), italic=True,
                                    units=self.config.get('text', 'units'),
                                    height=self.config.get('text', 'height'),
                                    wrapWidth=self.config.get('text', 'wrap_width')
                                    )
                )

        elif end_of_block:
            # guestimate the amount of money to be earned
            estimated_moneys = self.estimate_bonus()
            end_str = '!' if estimated_moneys > 0 else ''
            break_str = 'You can take a short break now. ' if not self.practice else ''

            self.current_instruction_screen = [
                visual.TextStim(self.screen, pos=(0, 4),
                                units=self.config.get('text', 'units'),
                                height=self.config.get('text', 'height'),
                                text='End of block. So far, you earned %d points%s' % (
                                self.total_points, end_str),
                                wrapWidth=self.config.get('text', 'wrap_width')
                                ),
                visual.TextStim(self.screen, pos=(0, 0),
                                units=self.config.get('text', 'units'),
                                height=self.config.get('text', 'height'),
                                text='Based on your performance so far, it looks like you will receive a bonus of approximately %.2f euro%s' % (estimated_moneys, end_str),
                                wrapWidth=self.config.get('text', 'wrap_width')
                                ),
                visual.TextStim(self.screen, pos=(0, -4),
                                units=self.config.get('text', 'units'),
                                height=self.config.get('text', 'height'),
                                text='%sPress <space bar> to continue.' % break_str,
                                wrapWidth=self.config.get('text', 'wrap_width')
                                )
            ]

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

    def prepare_objects(self, counterbalance=True):
        """
        Prepares all visual objects (instruction/feedback texts, stimuli)
        """

        config = self.config

        # Fixation cross
        self.fixation_cross = FixationCross(self.screen,
                                            outer_radius=config.get('fixation_cross', 'outer_radius'),
                                            inner_radius=config.get('fixation_cross', 'inner_radius'),
                                            bg=config.get('fixation_cross', 'bg'))

        # checkout if stimulus type is interpreted
        if not config.get('stimulus', 'type') in ['colors', 'agathodaimon']:
            raise(IOError('No idea what stimulus type I should draw. You entered %s' % config.get('stimulus', 'type')))
        # checkout if colors exist
        if config.get('stimulus', 'type') == 'colors':
            import matplotlib.colors as mcolors
            for set in ['set_1', 'set_2', 'set_3']:
                for col in config.get('stimulus', set):
                    if not col in mcolors.CSS4_COLORS.keys():
                        raise(IOError('I dont understand color %s that was provided to stimulus set %s...' %(col,
                                                                                                             set)))

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
                    screen=self.screen,
                    stimulus=stimulus,
                    stimulus_type=config.get('stimulus', 'type'),
                    width=config.get('stimulus', 'width'),
                    height=config.get('stimulus', 'height'),
                    text_height=config.get('stimulus', 'text_height'),
                    units=config.get('stimulus', 'units'),
                    x_pos=config.get('stimulus', 'x_pos')[i],
                    rect_line_width=config.get('stimulus',
                                               'rect_line_width'))

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
            visual.ImageStim(self.screen, image='./lib/vanilla.png'),
            visual.ImageStim(self.screen, image='./lib/sat.png')
        ]

        # Prepare feedback stimuli. Rendering of text is supposedly slow so better to do this once only (not every
        # trial)
        # 1. Feedback on outcomes
        self.feedback_outcome_objects = [
            visual.TextStim(win=self.screen, text='Outcome: 0',  color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[0])),
            visual.TextStim(win=self.screen, text='Outcome: +100', color='darkgreen',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[0])),
            visual.TextStim(win=self.screen, text='No choice made', color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[0]))
        ]
        # 2. Feedback on earnings
        self.feedback_earnings_objects = [
            visual.TextStim(win=self.screen, text='Reward: 0', color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos = (0, config.get('text', 'feedback_y_pos')[1])),
            visual.TextStim(win=self.screen, text='Reward: +100', color='darkgreen',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[1])),
        ]
        # 3. Feedback on timing
        self.feedback_timing_objects = [
            visual.TextStim(win=self.screen, text='Too slow!', color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[2])),
            visual.TextStim(win=self.screen, text='Too fast!', color='darkred',
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'),
                            pos=(0, config.get('text', 'feedback_y_pos')[2])),
        ]

        # Prepare cue texts. Rendering of text is supposedly slow so better to do this once only (not every
        # trial)
        self.cues = [
            # 0 = SPD
            visual.TextStim(win=self.screen, text="SPD",
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height')),

            # 1 = ACC
            visual.TextStim(win=self.screen, text="ACC",
                            units=config.get('text', 'units'),
                            height=config.get('text', 'height'))
        ]

        # Waiting for scanner screen
        self.scanner_wait_screen = visual.TextStim(win=self.screen,
                                                   text='Waiting for scanner...',
                                                   name='scanner_wait_screen',
                                                   units=config.get('text', 'units'), font='Helvetica Neue', pos=(0, 0),
                                                   italic=True,
                                                   height=config.get('text', 'height'), alignHoriz='center')

        if self.debug:
             pos = -config.get('screen', 'size')[0]/3, config.get('screen', 'size')[1]/3
             self.debug_txt = visual.TextStim(win=self.screen,
                                              alignVert='top',
                                              text='debug mode\n',
                                              name='debug_txt',
                                              units='pix',  # config.get('text', 'units'),
                                              font='Helvetica Neue',
                                              pos=pos,
                                              height=14,  # config.get('text', 'height'),
                                              alignHoriz='center')

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

        points = self.total_points
        max_points, bonus = self.estimate_bonus(return_final=True)
        print('Total points: %d (maximum: %d), total bonus earned: %.3f' %(points, max_points, bonus))

        self.save_data()
        super(LearningSession, self).close()

    def run_trial(self, ID, parameters, phase_durations, session, screen, annotate=False):

        if annotate:
            # adjust trial parameters to allow for annotation phases
            phase_2 = np.Inf if parameters['cue'] in ['SPD', 'ACC'] else -0.001
            phase_durations = np.array([-0.0001,   # phase 0: wait for scan pulse
                                        np.Inf,     # phase 1: fix cross
                                        phase_2,   # phase 2: cue
                                        -0.1,     # phase 3: fix cross
                                        np.Inf,   # phase 4: stimulus
                                        np.Inf,   # phase 5: choice highlight
                                        np.Inf,   # phase 6: feedback, highlight 1 (self-paced)
                                        np.Inf,   # phase 7: feedback, highlight 2 (self-paced)
                                        np.Inf,   # phase 8: feedback, highlight 3 (self-paced)
                                        0.5    # phase 9: ITI
                                        ])

            return PracticeTrial(ID, parameters, phase_durations, session=session, screen=screen)
        else:
            return LearningTrial(ID, parameters, phase_durations, session=session, screen=screen)

    def run(self):
        """ Runs this Instrumental Learning task"""
        # start by showing welcome screen, but only if the experiment is started from block 0
        if self.start_block == 1:
            self.update_instruction_screen(block=0, experiment_start=True)
            _ = InstructionTrial(ID=self.instruction_trial_n,
                                 parameters={},
                                 phase_durations=[0.5, 1000],
                                 session=self,
                                 screen=self.screen).run()
            self.instruction_trial_n -= 1

        # # Set start time of block 0 to 0 (useful if you want to calculate durations on the fly, otherwise not so
        # important
        # self.block_start_time = 0


        all_blocks = np.unique(self.design.block)
        current_block_id = -1
        while True:   # while loop to allow for moving between blocks
            current_block_id += 1
            if len(all_blocks) <= current_block_id:
                print('End of experiment!')
                # self.stopped = True
                break
            block_n = all_blocks[current_block_id]

            if block_n < self.start_block:
                continue

            # Get all trials for the next block, create trial handler
            this_block_design = self.design.loc[self.design.block == block_n]
            trial_handler = data.TrialHandler(this_block_design.to_dict('records'),
                                              nReps=1,
                                              method='sequential')

            # Loop over trials
            for block_trial_ID, this_trial_info in enumerate(trial_handler):

                # show instruction screen (do this in inner loop so we can peek into the next cue)
                if block_n in [1, 4] and block_trial_ID == 0:
                    if this_trial_info['cue'] in ['SPD', 'ACC']:
                        self.update_instruction_screen(block=block_n, task='SAT', experiment_start=False)
                    else:
                        self.update_instruction_screen(block=block_n, task='vanilla', experiment_start=False)
                    _ = InstructionTrial(ID=self.instruction_trial_n,
                                         parameters={},
                                         phase_durations=[0.5, 1000],
                                         session=self,
                                         screen=self.screen).run()
                    self.instruction_trial_n -= 1

                    # also, show upcoming stimuli
                    self.update_instruction_screen(block=block_n, show_upcoming_stimuli=True)
                    _ = InstructionTrial(ID=self.instruction_trial_n,
                                         parameters={},
                                         phase_durations=[0.5, 1000],
                                         session=self,
                                         screen=self.screen).run()
                    self.instruction_trial_n -= 1

                # Actual trial
                this_trial_parameters = {'stim_left': this_trial_info['stim_left'],
                                         'stim_right': this_trial_info['stim_right'],
                                         'correct_response': this_trial_info['correct_stim_lr'],
                                         'block': block_n,
                                         'block_trial_ID': block_trial_ID,
                                         'p_win': [this_trial_info['p_win_left'], this_trial_info['p_win_right']],
                                         'cue': this_trial_info['cue']}

                these_phase_durations = self.phase_durations.copy()
                for phase_n in np.arange(8):
                    if 'phase_' + str(phase_n) in this_trial_info.keys():
                        these_phase_durations[phase_n] = this_trial_info['phase_' + str(phase_n)]

                # NB we stop the trial 0.5s before the start of the new trial, to allow sufficient computation time
                # for preparing the next trial. (but never below 0.1s)
                these_phase_durations[-1] = np.max([0.1, these_phase_durations[-1]-0.5])

                this_trial = self.run_trial(ID=int(this_trial_info.trial_ID),
                                            annotate=this_trial_info.annotate,
                                            parameters=this_trial_parameters,
                                            phase_durations=these_phase_durations,
                                            session=self,
                                            screen=self.screen)

                # run the prepared trial
                this_trial.run()

                # Record some stuff
                trial_handler.addData('rt', this_trial.response['rt'])
                trial_handler.addData('response', this_trial.response['button'])
                trial_handler.addData('deadline', this_trial.deadline)

                trial_handler.addData('points_earned', this_trial.points_earned)
                trial_handler.addData('outcome', this_trial.won)
                self.total_points += this_trial.points_earned
                self.total_trials += 1
                trial_handler.addData('total_points_earned', self.total_points)

                # absolute times since session start
                for time_name in ['start_time', 't_time', 'jitter_time_1', 'cue_time', 'jitter_time_2',
                                  'stimulus_time', 'selection_time', 'feedback_time', 'iti_time']:
                    trial_handler.addData(time_name, getattr(this_trial, time_name))

                # durations / time since actual start of the block. These are useful to create events-files later for
                #  convolving. Can also grab these from the eventArray though.
                # trial_handler.addData('trial_t_time_block_measured', this_trial.t_time - self.block_start_time)
                # trial_handler.addData('stimulus_onset_time_block_measured', this_trial.jitter_time -
                #                       self.block_start_time)
                # Counter-intuitive, but jitter_time is END of the jitter period = onset of stim

                if self.restart_block:
                    current_block_id -= 1
                    break

                # check for last trial?
                # if this_trial_info.trial_ID == self.design.trial_ID.max():
                #     self.stopped = True

                if self.stopped:
                    # out of trial
                    break

            # Save
            self.save_data(trial_handler, block_n)

            if self.stopped:
                # out of block
                break

            if not self.restart_block:
                if not self.stopped:
                    # end of block, show expected score
                    self.update_instruction_screen(end_of_block=True)
                    _ = InstructionTrial(ID=self.instruction_trial_n,
                                         parameters={},
                                         phase_durations=[0.5, 1000],
                                         session=self,
                                         screen=self.screen).run()
                    self.instruction_trial_n -= 1
            else:
                self.restart_block = False

        if not self.stopped:
            # End of experiment
            print('yes?')
            self.update_instruction_screen(end_of_session=True)
            _ = InstructionTrial(ID=self.instruction_trial_n,
                                 parameters={},
                                 phase_durations=[0.5, 1000],
                                 session=self,
                                 screen=self.screen).run()
            self.instruction_trial_n -= 1

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

    # sub_id = 'PRACTICE'
    # Set-up session
    sess = LearningSession('1',
                           1,
                           tr=0,
                           start_block=0,
                           config=config,
                           debug=True,
                           practice=True)

    # EMULATOR
    # from psychopy.hardware.emulator import launchScan
    # scanner_emulator = launchScan(win=sess.screen, settings={'TR': 0.5, 'volumes': 30000, 'sync': 't'}, mode='Test')

    # run
    sess.run()

    # Load & dump data
    import cPickle as pkl
    from pprint import pprint

    with open(sess.output_file + '_outputDict.pkl', 'r') as f:
        a = pkl.load(f)
    pprint(a)

    core.quit()