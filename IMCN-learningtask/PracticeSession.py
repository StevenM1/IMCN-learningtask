from exptools2.core.session import Session
from LearningTrial import LearningTrial, InstructionTrial, TextTrial, CheckTrial, PracticeTrial
from LearningStimulus import FixationCross, LearningStimulusSingle
from psychopy import visual
from psychopy import core
from psychopy.visual import TextStim
from psychopy.event import waitKeys
from LearningSession import LearningSession
import numpy as np
import pandas as pd
import os
import os.path as op
import sys


class PracticeSession(LearningSession):

    def __init__(self, index_number, output_str, output_dir, settings_file,
                 start_block):
        super(PracticeSession, self).__init__(index_number=index_number,
                                              start_block=start_block,
                                              output_str=output_str + '_practice',
                                              output_dir=output_dir,
                                              settings_file=settings_file,
                                              debug=False,
                                              scanner=False)

        self.start_location = start_block
        # self.index_number = index_number  # participant number
        # self.start_block = start_block    # allows for starting at a later block than 1
        # self.debug = debug
        # self.in_scanner = scanner
        # self.design = None
        # self.p_wins = None
        # self.trials = None
        # self.stimuli = None
        # self.fixation_cross = None
        # self.total_points = 0
        # self.total_trials = 0
        #
        # self.response_button_signs = [self.settings['input']['response_button_left'],
        #                               self.settings['input']['response_button_right']]
        #
        # # note that all these durations are default values and can and should be overwritten before each trial
        # self.phase_durations = np.array([1,     # phase 0: fix cross1 (ITI pretrial)
        #                                  1,     # phase 1: cue
        #                                  1,     # phase 2: fix cross2
        #                                  2,     # phase 3: stimulus
        #                                  0.5,   # phase 4: choice highlight
        #                                  0.5,   # phase 5: feedback
        #                                  1      # phase 6: fix cross3 (ITI posttrial)
        #                                  ])
        # self.phase_names = ['fix_cross_1', 'cue', 'fix_cross_2', 'stimulus', 'highlight', 'feedback', 'iti']

    def prepare_objects(self):
        """
        Prepares all visual objects (instruction/feedback texts, stimuli)
        """

        settings = self.settings

        self.continue_instruction = TextStim(self.win, text='Press <space bar> to continue',
                                             italic=True, pos=(0, -6))
        self.back_instruction = TextStim(self.win, text='Press <backspace> to go back',
                                         italic=True, pos=(0, -7))

        all_symbols = ['a', 'b', 'c', 'd', 'x', 'y']
        self.design = pd.DataFrame({'stim_high': all_symbols,
                                    'stim_low': all_symbols})

        super(PracticeSession, self).prepare_objects()

    def display_wait_text(self, text, stims_to_draw, text_per_line=None, keys=None, duration=None, **kwargs):
        """ Displays a slightly more complex text """

        if text_per_line is not None:
            for text_nr, text_ in enumerate(text_per_line):
                stim = TextStim(win=self.win, text=text_, pos=(0, 3-text_nr), wrapWidth=500, units='deg',
                                alignVert='top')
                stims_to_draw.append(stim)

        for stim in stims_to_draw:
            stim.draw()

        self.display_text(text=text, keys=keys, duration=duration, **kwargs)

    def generate_text_objects(self, text_per_line, degrees_per_line=1, bottom_pos=3,
                              wrapWidth=100, **kwargs):

        text_objects = []
        for i, text in enumerate(text_per_line):
            text_objects.append(visual.TextStim(self.win, text=text, pos=(0, bottom_pos-i*degrees_per_line),
                                                alignVert='bottom', wrapWidth=wrapWidth, **kwargs))

        return text_objects

    def get_random_cue_location(self, symbols, ps):

        stim_out = [symbols[0], symbols[1]]
        p_out = [ps[0], ps[1]]
        cue_out = 'SPD'
        if np.random.sample() < .5:
            stim_out = [symbols[1], symbols[0]]
            p_out = [ps[1], ps[0]]
        if np.random.sample() < .5:
            cue_out = 'ACC'

        return stim_out, p_out, cue_out

    def get_jittered_durations(self):

        durations = np.zeros(7)
        durations[0] = np.random.choice([.5, 1.25, 2])
        durations[1] = 1 + np.random.choice([.5, 1.25, 2])
        durations[2] = np.random.choice([.5, 1.25, 2])
        durations[3] = 1.5 + np.random.choice([0, .5, 1.25])
        durations[4] = 1 + np.random.choice([0, .5, 1.25])
        durations[5] = 1 + np.random.choice([0, .5, 1.25])
        durations[6] = np.random.choice([0, .5, 1.25])

        return durations

    def run(self):
        """ Runs this Instrumental Learning task"""

        # self.load_design()
        self.prepare_objects()
        self.start_experiment()

        self.display_wait_text(text='Welcome to this practice session!', keys='space',
                               stims_to_draw=[self.continue_instruction], pos=(0, 3))

        self.display_wait_text(text=' ',
                               text_per_line=['It is important that you understand the task well.',
                                              'Please read these instructions carefully, ',
                                              'and do not hesitate to ask any questions if anything is unclear'],
                               keys='space',
                               stims_to_draw=[self.continue_instruction], pos=(0, 3))

        continue_back = [self.continue_instruction, self.back_instruction]
        continue_only = [self.continue_instruction]
        loop_1_done = False
        loop_2_done = False
        answer_1_checked = False
        answer_2_checked = False
        session_location = self.start_location
        trial_nr = 0
        while True:
            if session_location == 0:
                next_texts = self.generate_text_objects(['You will see two choice options, such as these'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left']['a'], self.stimuli['right']['b']] +
                                                  next_texts + continue_only,
                               session=self)
                tr.run()
            elif session_location == 1:
                next_texts = self.generate_text_objects(['These act like slot machines in a casino',
                                                         'On every attempt, you have a chance to get a reward'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left']['a'], self.stimuli['right']['b']] +
                                                  next_texts + continue_back,
                               session=self)
                tr.run()
            elif session_location == 2:
                next_texts = self.generate_text_objects(
                    ['The probability of getting a reward differs per symbol',
                     'You need to discover, by trial and error, which symbol',
                     'has a higher chance of giving a reward'],
                    bottom_pos=4)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left']['a'], self.stimuli['right']['b']] +
                                                  next_texts + continue_back,
                               session=self)
                tr.run()
            elif session_location == 3:
                next_texts = self.generate_text_objects(
                    ['Let\'s practice this for a bit to give you a feeling how this works!',
                     'Make a couple choices by pressing <z> for left or <m> for right',
                     'After every choice, you will see whether you got a reward (+100) or '
                     'not (+0)'],
                    bottom_pos=5)
                move_on_text = [
                    visual.TextStim(win=self.win, text='When you think you know which symbol has the highest '
                                                       'probability of giving a reward, press <space bar>',
                                    italic=True, pos=(0, -6), wrapWidth=100)]

                if not loop_1_done:
                    for _ in range(10):
                        tr = InstructionTrial(trial_nr=trial_nr,
                                              parameters={'cue': '',  # for compatibility
                                                          'block_nr': session_location,  # for compatibility
                                                          'stimulus_symbol_left': 'a',
                                                          'stimulus_symbol_right': 'b',
                                                          'p_win_left': 0.8,
                                                          'p_win_right': 0.2},
                                              phase_durations=[0.000, 0.000, 0.000, 60, 1, 1, 0],
                                              phase_names=self.phase_names,
                                              session=self,
                                              decoration_objects=next_texts)
                        tr.run()
                        trial_nr += 1
                    loop_1_done = True

                while True:
                    tr = InstructionTrial(trial_nr=trial_nr,
                                          parameters={'cue': '',  # for compatibility
                                                      'block_nr': session_location,  # for compatibility
                                                      'stimulus_symbol_left': 'a',
                                                      'stimulus_symbol_right': 'b',
                                                      'p_win_left': 0.8,
                                                      'p_win_right': 0.2},
                                          phase_durations=[0.000, 0.000, 0.000, 60, 1, 1, 0],
                                          phase_names=self.phase_names,
                                          session=self,
                                          decoration_objects=next_texts + move_on_text,
                                          allow_space_break=True)
                    tr.run()
                    trial_nr += 1

                    if tr.last_key is not None:
                        print('Not none found')
                        if tr.last_key == 'space':
                            print('I should break')
                            break

                # check correct here, give feedback
                if not answer_1_checked:
                    next_texts = self.generate_text_objects(
                        ['Which symbol is more likely to give a reward?'],
                        bottom_pos=5)
                    tr = CheckTrial(trial_nr=trial_nr,
                                    parameters={'cue': '',  # for compatibility
                                                'block_nr': session_location,  # for compatibility
                                                'stimulus_symbol_left': 'a',
                                                'stimulus_symbol_right': 'b',
                                                'p_win_left': 0.8,
                                                'p_win_right': 0.2},
                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 60, 0],
                                    phase_names=self.phase_names,
                                    session=self,
                                    decoration_objects=next_texts,
                                    allow_space_break=True)
                    tr.run()
                    answer_1_checked = True
                    print(tr.last_key)
                    # tr.last_key = 'space'

            elif session_location == 4:
                next_texts = self.generate_text_objects(
                    ['It\'s important to realise that:',
                     '- The low probability symbol *sometimes* gives a reward,\n'
                     'so you could be wrong even if you got a reward for a choice',
                     '- The high probability symbol *not always* gives a reward,\n'
                     'so you could be right even if you did not get a reward for a choice',
                     'You will thus need to balance between committing to one symbol, '
                     'and exploring both symbols'],
                    bottom_pos=5, degrees_per_line=3)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_only,
                               session=self)
                tr.run()

            elif session_location == 5:
                next_texts = self.generate_text_objects(
                    ['Let\'s try again with new symbols. It\'s more difficult now',
                     'and the location of the symbols will alternate randomly',
                     'Make a couple of decisions by pressing <z> for left or <m> for right'],
                    bottom_pos=5)

                if not loop_2_done:
                    for _ in range(10):
                        symbols, ps, _ = self.get_random_cue_location(ps=[0.65, 0.35], symbols=['c', 'd'])
                        tr = InstructionTrial(trial_nr=trial_nr,
                                              parameters={'cue': '',      # for compatibility
                                                          'block_nr': session_location,  # for compatibility
                                                          'stimulus_symbol_left': symbols[0],
                                                          'stimulus_symbol_right': symbols[1],
                                                          'p_win_left': ps[0],
                                                          'p_win_right': ps[1]},
                                              phase_durations=[0.000, 0.000, 0.000, 60, 1, 1, 0],
                                              phase_names=self.phase_names,
                                              session=self,
                                              decoration_objects=next_texts)
                        tr.run()
                        trial_nr += 1

                    loop_2_done = True

                while True:
                    symbols, ps, _ = self.get_random_cue_location(ps=[0.65, 0.35], symbols=['c', 'd'])
                    tr = InstructionTrial(trial_nr=trial_nr,
                                          parameters={'cue': '',  # for compatibility
                                                      'block_nr': session_location,
                                                      'stimulus_symbol_left': symbols[0],
                                                      'stimulus_symbol_right': symbols[1],
                                                      'p_win_left': ps[0],
                                                      'p_win_right': ps[1]},
                                          phase_durations=[0.000, 0.000, 0.000, 60, 1, 1, 0],
                                          phase_names=self.phase_names,
                                          session=self,
                                          decoration_objects=next_texts + move_on_text,
                                          allow_space_break=True)
                    tr.run()
                    trial_nr += 1

                    if tr.last_key is not None:
                        print('Not none found')
                        if tr.last_key == 'space':
                            print('I should break')
                            break

                # check correct here, give feedback
                if not answer_2_checked:
                    next_texts = self.generate_text_objects(
                        ['Which symbol is more likely to give a reward??'],
                        bottom_pos=5)
                    tr = CheckTrial(trial_nr=trial_nr,
                                    parameters={'cue': '',      # for compatibility
                                                'block_nr': session_location,  # for compatibility
                                                'stimulus_symbol_left': 'c',
                                                'stimulus_symbol_right': 'd',
                                                'p_win_left': 0.8,
                                                'p_win_right': 0.2},
                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 60, 0],
                                    phase_names=self.phase_names,
                                    session=self,
                                    decoration_objects=next_texts,
                                    allow_space_break=True)
                    tr.run()
                    answer_2_checked = True
                    print(tr.last_key)
                    # tr.last_key = 'space'

            elif session_location == 6:
                next_texts = self.generate_text_objects(
                    ['Alright, well done so far!',
                     'The next aspect of the task are the *cues*'],
                    bottom_pos=5, degrees_per_line=3)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_only,
                               session=self)
                tr.run()

            elif session_location == 7:
                next_texts = self.generate_text_objects(
                    ['Cues instruct you, before each trial, how fast you should respond',
                     '',
                     'If you see "SPD" (for \'speed\'), you need to respond *quickly*:',
                     'You need to respond fast, even if you make some more mistakes in your choices',
                     '',
                     'If you see "ACC" (for \'accuracy\'), you need to respond *accurately*:',
                     'You need to make the correct choice, even if that takes a little bit longer'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif session_location == 8:
                next_texts = self.generate_text_objects(
                    ['Following the SPD / ACC cues is important!',
                     'If you see "SPD" but respond too slowly, you *lose* points',
                     'Only if you respond fast, you earn the outcome of the choice (+100 or +0)',
                     '',
                     'If you see "ACC" but make an incorrect response, you are less likely to earn points',
                     'Taking a bit more time to make the correct choice will lead to more points'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif session_location == 9:
                next_texts = self.generate_text_objects(
                    ['Let\'s try and see how this works!'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif session_location == 10:

                tr = PracticeTrial(trial_nr=trial_nr,
                                   parameters={'cue': 'ACC',
                                               'block_nr': session_location,
                                               'stimulus_symbol_left': 'x',
                                               'stimulus_symbol_right': 'y',
                                               'p_win_left': 1,
                                               'p_win_right': 1},
                                   phase_durations=[60, 60, 0, 60, 60, 60, 0, 60, 60],
                                   phase_names=self.phase_names + ['feedback_2', 'feedback_3'],
                                   # decoration_objects=next_texts + continue_back,
                                   session=self)
                tr.run()

            elif session_location == 11:
                next_texts = self.generate_text_objects(
                    ['Let\'s now practice this for real! When you press space bar,',
                     'a short practice block of 15 trials starts'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()
                trial_nr += 1

                for i in range(15):
                    symbols, ps, cue = self.get_random_cue_location(ps=[0.75, 0.25], symbols=['x', 'y'])

                    tr = LearningTrial(trial_nr=trial_nr,
                                       parameters={'cue': cue,
                                                   'block_nr': session_location,
                                                   'stimulus_symbol_left': symbols[0],
                                                   'stimulus_symbol_right': symbols[1],
                                                   'p_win_left': ps[0],
                                                   'p_win_right': ps[1]},
                                       phase_durations=self.get_jittered_durations(),
                                       phase_names=self.phase_names,
                                       session=self)
                    tr.run()
                    trial_nr += 1

            trial_nr += 1
            if tr.last_key == 'space':
                session_location += 1
            elif tr.last_key == 'backspace':
                session_location -= 1

            if session_location == 12:
                break

        self.close()
        self.quit()







        # # remove blocks before start_block
        # all_blocks = np.unique(self.design.block)
        # all_blocks = all_blocks[self.start_block-1:]  # assuming start_block is 1-coded
        #
        # for block_nr in all_blocks:
        #     self.create_trials(block_nr)
        #
        #     if self.exp_start is None:
        #         self.start_experiment()
        #
        #     self.display_text('Waiting for scanner', keys=self.mri_trigger)
        #     self.timer.reset()
        #
        #     # loop over trials
        #     for trial in self.trials:
        #         trial.run()
        #
        #     # save data
        #     self.save_data(block_nr=block_nr)

        self.close()
        self.quit()


if __name__ == '__main__':



    import datetime
    index_number = 1
    start_block = 10
    scanner = False
    simulate = 'y'
    debug = True

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
    output_str = f'sub-{index_number}_task-practice-learning_datetime-{timestamp}'
    output_dir = './data'
    if simulate == 'y':
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings_simulate.yml'
    else:
        settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings.yml'

    sess = PracticeSession(index_number=index_number,
                           output_str=output_str,
                           output_dir=output_dir,
                           settings_file=settings_file,
                           start_block=start_block)
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