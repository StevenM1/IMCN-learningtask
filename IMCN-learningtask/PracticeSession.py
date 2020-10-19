from LearningTrial import LearningTrial, InstructionTrial, TextTrial, CheckTrial, AnnotatedTrial
from psychopy import visual
from psychopy.visual import TextStim
from LearningSession import LearningSession
import numpy as np
import pandas as pd


class PracticeSession(LearningSession):
    """
    This code introduces the instrumental learning task and gives participants the chance to practice.
    It shows up to 20 different little "blocks", which range in complexity from showing a single screen to practicing
    15 trials.

    All screens:
    0: "Welcome to this practice session"
    1: "Important to read instructions carefully"
    2: "You will see two choice options"
    3: "These act like slot machines in a casino"
    4: "The probability of getting a reward differs per symbol"
    5: Practice single symbol 1
    6: "It\'s important to realise that:"
    7: "Let's practice again"
    8: Introduce differing symbols
    9: Practice 15 trials with differing symbols

    ------ Screens below introduce the SAT cues
    10: 'Alright, well done so far!' 'The final aspect of the task are the *cues*'
    11: 'Cues instruct you to ...'
    12: 'Following the SPD / ACC cues is important!'
    13: Let's see how this works
    14: Slow example trial / "video"
    15: Let's practice for real, 15 trials
    16: You're ready to start SAT

    ------ Screens below re-cap the task, in case the task order is SAT - reversal
    17: Next, you will again perform the symbol-learning task. This time, there will be no cues. Always respond as fast
    as possible without making mistakes. Just as a recap, you will now practice 15 trials again.
    18: You're ready for the Reversal task

    ------ Screen below recaps the task, in case the task order is reversal - SAT
    19: Next, you will again perform the symbol-learning task. However, it will be a bit more complex than before,
    since you will be presented with *cues*.

    """
    def __init__(self, output_str, output_dir, settings_file,
                 start_block, practice_n = 1, SAT_first=True):
        super(PracticeSession, self).__init__(start_block=start_block,
                                              output_str=output_str + '_practice',
                                              output_dir=output_dir,
                                              settings_file=settings_file,
                                              debug=False,
                                              scanner=False)
        self.practice_n = practice_n
        self.SAT_first = SAT_first

        if self.practice_n == 1:
            if self.SAT_first:
                self.presentation_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            else:
                self.presentation_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18]

        elif self.practice_n == 2:
            if self.SAT_first:
                self.presentation_order = [17, 9, 18]
            else:
                self.presentation_order = [19, 11, 12, 13, 14, 15, 16]

        self.response_button_signs = [self.settings['input']['response_button_left'],
                                      self.settings['input']['response_button_right']]

        self.stop = False
        self.session_location_idx = self.start_block
        self.session_location = self.presentation_order[self.session_location_idx]
        self.show_timing_feedback = False

    def prepare_objects(self):
        """
        Prepares all visual objects (instruction/feedback texts, stimuli)
        """

        self.continue_instruction = TextStim(self.win, text='Press <space bar> to continue',
                                             italic=True, pos=(0, -6))
        self.back_instruction = TextStim(self.win, text='Press <backspace> to go back',
                                         italic=True, pos=(0, -7))

        # this is an ugly workaround since the method's parent assumes there's a self.design with stim_high and _low
        # cols
        self.all_symbols = ['A', 'H', 'W', 'l', 'Z', 'm', 'b', 'w', 'L', 'r', 'G', 'z']
        self.design = pd.DataFrame({'stim_high': self.all_symbols,
                                    'stim_low': self.all_symbols})

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

    def get_random_stimuli_ps(self, stimuli=None, ps=((.8, .2), (.7, .3), (.6, .4))):

        if stimuli is None:
            stimuli = ((self.all_symbols[-1], self.all_symbols[-2]),
                       (self.all_symbols[-3], self.all_symbols[-4]),
                       (self.all_symbols[-5], self.all_symbols[-6]))

        choice = np.random.randint(low=0, high=len(stimuli), size=1)[0]
        return stimuli[choice], ps[choice]

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

    def get_jittered_durations(self, include_cue=True):
        """ Generates some simple, jittered durations """

        durations = np.zeros(9)
        durations[0] = np.random.choice([0.5, 1, 1.5, 2])     # pre-cue
        durations[1] = 1 if include_cue else 0                # cue
        durations[2] = np.random.choice([0.5, 1, 1.5, 2])     # jitter cue-stimulus
        durations[3] = 1.5                                    # stimulus
        durations[4] = np.random.choice([0.5, 1, 1.5, 2])     # jitter stim-highlight
        durations[5] = 0 #1                                   # highlight
        durations[6] = 0  # np.random.choice([0, 1, 1.5])     # jitter highlight-feedback
        durations[7] = 1                                      # feedback
        durations[8] = 0  # np.random.choice([0, 1, 1.5])     # iti

        return durations

    def move_forward(self):
        if self.session_location_idx == (len(self.presentation_order)-1):
            # can't move forward, end is reached
            self.stop = True
            return 0

        self.session_location_idx += 1  # np.min(len(self.presentation_order), self.session_location_idx+1)
        self.session_location = self.presentation_order[self.session_location_idx]

    def move_backward(self):
        self.session_location_idx = np.max([0, self.session_location_idx-1])  # prevent moving below 0
        self.session_location = self.presentation_order[self.session_location_idx]

    def run(self, quit_on_exit=False):
        """ Runs this Instrumental Learning practice session """

        self.prepare_objects()
        self.start_experiment()

        #
        # self.display_wait_text(text='Welcome to this practice session!', keys='space',
        #                        stims_to_draw=[self.continue_instruction], pos=(0, 3))
        #
        # self.display_wait_text(text=' ',
        #                        text_per_line=['It is important that you understand the task well.',
        #                                       'Please read these instructions carefully, ',
        #                                       'and do not hesitate to ask any questions if anything is unclear'],
        #                        keys='space',
        #                        stims_to_draw=[self.continue_instruction], pos=(0, 3))

        continue_back = [self.continue_instruction, self.back_instruction]
        continue_only = [self.continue_instruction]
        loop_1_done = False
        loop_2_done = False
        answer_1_checked = False
        answer_2_checked = False
        trial_nr = 0

        # start main loop
        while True:
            if self.session_location == 0:
                next_texts = self.generate_text_objects(['Welcome to this practice session!'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[300],
                               decoration_objects=next_texts + continue_only,
                               session=self)
                tr.run()

            elif self.session_location == 1:
                next_texts = self.generate_text_objects([
                                              'It is important that you understand the task well.',
                                              'Please read these instructions carefully, ',
                                              'and do not hesitate to ask any questions if anything is unclear'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[300],
                               decoration_objects=next_texts + continue_only,
                               session=self)
                tr.run()

            elif self.session_location == 2:
                next_texts = self.generate_text_objects(['You will see two choice options, such as these'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left'][self.all_symbols[0]],
                                                   self.stimuli['right'][self.all_symbols[1]]] +
                                                   next_texts + continue_only,
                               session=self)
                tr.run()

            elif self.session_location == 3:
                next_texts = self.generate_text_objects(['These act like slot machines in a casino',
                                                         'On every attempt, you have a chance to get a reward'])
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left'][self.all_symbols[0]],
                                                   self.stimuli['right'][self.all_symbols[1]]] +
                                                   next_texts + continue_back,
                               session=self)
                tr.run()

            elif self.session_location == 4:
                next_texts = self.generate_text_objects(
                    ['The probability of getting a reward differs per symbol',
                     'You need to discover, by trial and error, which symbol',
                     'has a higher chance of giving a reward'],
                    bottom_pos=4)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=[self.stimuli['left'][self.all_symbols[0]],
                                                   self.stimuli['right'][self.all_symbols[1]]] +
                                                  next_texts + continue_back,
                               session=self)
                tr.run()

            elif self.session_location == 5:
                next_texts = self.generate_text_objects(
                    ['Let\'s practice this for a bit to give you a feeling how this works!',
                     'Make a couple choices by pressing <{}> for left or <{}> for right'.format(
                         self.response_button_signs[0], self.response_button_signs[1])  ,
                     'After every choice, you will see whether you got a reward (+100) or '
                     'not (+0)'],
                    bottom_pos=5)
                choose_now_txt = [
                    visual.TextStim(win=self.win, text='Make a choice now',
                                    italic=True, pos=(0, -6), wrapWidth=100)]
                move_on_text = [
                    visual.TextStim(win=self.win, text='When you think you know which symbol has the highest '
                                                       'probability of giving a reward, press <space bar>',
                                    italic=True, pos=(0, -7), wrapWidth=100)]

                if not loop_1_done:
                    for _ in range(10):
                        tr = InstructionTrial(trial_nr=trial_nr,
                                              parameters={'cue': '',  # for compatibility
                                                          'block_nr': self.session_location,  # for compatibility
                                                          'stimulus_symbol_left': self.all_symbols[0],
                                                          'stimulus_symbol_right': self.all_symbols[1],
                                                          'p_win_left': 0.8,
                                                          'p_win_right': 0.2},
#                                              phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 1, 0],
                                              phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 1, 0],
                                              phase_names=self.phase_names,
                                              session=self,
                                              decoration_objects=next_texts + choose_now_txt)
                        tr.run()
                        trial_nr += 1
                    loop_1_done = True

                while True:
                    tr = InstructionTrial(trial_nr=trial_nr,
                                          parameters={'cue': '',  # for compatibility
                                                      'block_nr': self.session_location,  # for compatibility
                                                      'stimulus_symbol_left': self.all_symbols[0],
                                                      'stimulus_symbol_right': self.all_symbols[1],
                                                      'p_win_left': 0.8,
                                                      'p_win_right': 0.2},
#                                          phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 1, 0],
                                          phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 1, 0],
                                          phase_names=self.phase_names,
                                          session=self,
                                          decoration_objects=next_texts + move_on_text + choose_now_txt)
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
                                                'block_nr': self.session_location,  # for compatibility
                                                'stimulus_symbol_left': self.all_symbols[0],
                                                'stimulus_symbol_right': self.all_symbols[1],
                                                'p_win_left': 0.8,
                                                'p_win_right': 0.2},
#                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 60, 0],
                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 60, 0],
                                    phase_names=self.phase_names,
                                    session=self,
                                    decoration_objects=next_texts,
                                    break_keys=['space', 'backspace'])
                    tr.run()
                    answer_1_checked = True
                    if tr.last_key == 'backspace':
                        self.session_location_idx += 1  # add one, which will be subtracted at the end of the outer
                        # while loop

            elif self.session_location == 6:
                next_texts = self.generate_text_objects(
                    ['It\'s important to realise that:',
                     '- The low probability symbol *sometimes* gives a reward,\n'
                     'so you could be wrong even if you got a reward for a choice',
                     '- The high probability symbol does *not always* give a reward,\n'
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

            elif self.session_location == 7:
                next_texts = self.generate_text_objects(
                    ['Let\'s try again with new symbols. It\'s more difficult now',
                     'and the location of the symbols will alternate randomly',
                     'Make a couple of decisions by pressing <{}> for left or <{}> for right'.format(
                         self.response_button_signs[0], self.response_button_signs[1])],
                    bottom_pos=5)
                choose_now_txt = [
                    visual.TextStim(win=self.win, text='Make a choice now',
                                    italic=True, pos=(0, -6), wrapWidth=100)]

                if not loop_2_done:
                    for _ in range(10):
                        symbols, ps, _ = self.get_random_cue_location(ps=[0.65, 0.35],
                                                                      symbols=[self.all_symbols[2],
                                                                               self.all_symbols[3]])
                        tr = InstructionTrial(trial_nr=trial_nr,
                                              parameters={'cue': '',      # for compatibility
                                                          'block_nr': self.session_location,  # for compatibility
                                                          'stimulus_symbol_left': symbols[0],
                                                          'stimulus_symbol_right': symbols[1],
                                                          'p_win_left': ps[0],
                                                          'p_win_right': ps[1]},
#                                              phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 1, 0],
                                              phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 1, 0],
                                              phase_names=self.phase_names,
                                              session=self,
                                              decoration_objects=next_texts + choose_now_txt)
                        tr.run()
                        trial_nr += 1

                    loop_2_done = True

                while True:
                    # symbols, ps = self.get_random_stimuli_ps()
                    symbols, ps, _ = self.get_random_cue_location(ps=[0.65, 0.35],
                                                                  symbols=[self.all_symbols[2],
                                                                           self.all_symbols[3]])
                    tr = InstructionTrial(trial_nr=trial_nr,
                                          parameters={'cue': '',  # for compatibility
                                                      'block_nr': self.session_location,
                                                      'stimulus_symbol_left': symbols[0],
                                                      'stimulus_symbol_right': symbols[1],
                                                      'p_win_left': ps[0],
                                                      'p_win_right': ps[1]},
#                                          phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 1, 0],
                                          phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 1, 0],
                                          phase_names=self.phase_names,
                                          session=self,
                                          decoration_objects=next_texts + move_on_text + choose_now_txt)
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
                        ['Which symbol is more likely to give a reward?'],
                        bottom_pos=5)
                    tr = CheckTrial(trial_nr=trial_nr,
                                    parameters={'cue': '',      # for compatibility
                                                'block_nr': self.session_location,  # for compatibility
                                                'stimulus_symbol_left': self.all_symbols[2],
                                                'stimulus_symbol_right': self.all_symbols[3],
                                                'p_win_left': 0.8,
                                                'p_win_right': 0.2},
#                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 1, 0, 60, 0],
                                    phase_durations=[0.000, 0.000, 0.000, 60, 0, 0, 0, 60, 0],
                                    phase_names=self.phase_names,
                                    session=self,
                                    decoration_objects=next_texts)
                    tr.run()
                    answer_2_checked = True
                    if tr.last_key == 'backspace':
                        self.session_location_idx += 1  # add one, which will be subtracted at the end of the outer
                        # while loop

            elif self.session_location == 8:
                next_texts = self.generate_text_objects(
                    ['Well done!',
                     '',
                     'The experiment consists of multiple \'blocks\'',
                     'Within each block, you will see *three* different pairs of symbols',
                     'that you need to choose between', '',
                     'These pairs of symbols alternate between trials',
                     '',
                     'In every new block, you will see *new* pairs of symbols',
                     'and you need to learn these from scratch'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif self.session_location == 9:
                next_texts = self.generate_text_objects(
                    ['You will next practice a short block of trials with alternating symbols',
                     '',
                     'Note that the options do not immediately disappear when you press a button. This is normal.'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

                trial_nr += 1
                ## ugly: disable timing feedback in this block
                for i in range(15):
                    symbols, ps = self.get_random_stimuli_ps()
                    symbols, ps, cue = self.get_random_cue_location(ps=ps, symbols=symbols)

                    tr = LearningTrial(trial_nr=trial_nr,
                                       parameters={'cue': 'ACC',
                                                   'block_nr': self.session_location,
                                                   'stimulus_symbol_left': symbols[0],
                                                   'stimulus_symbol_right': symbols[1],
                                                   'p_win_left': ps[0],
                                                   'p_win_right': ps[1],
                                                   'n_trs': 5},
                                       phase_durations=self.get_jittered_durations(include_cue=False),
                                       phase_names=self.phase_names,
                                       session=self)
                    tr.run()
                    tr.last_key = tr.last_resp  # for compatibility
                    trial_nr += 1

                self.move_forward()
                #session_location += 1  # manually forward here, since last key is not 'space'

            elif self.session_location == 10:
                next_texts = self.generate_text_objects(
                    ['Alright, well done so far!',
                     'The final aspect of the task are the *cues*'],
                    bottom_pos=5, degrees_per_line=3)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_only,
                               session=self)
                tr.run()

            elif self.session_location == 11:
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

            elif self.session_location == 12:
                next_texts = self.generate_text_objects(
                    ['Following the SPD / ACC cues is important!', '',
                     'If you see "SPD" but respond too slowly, you *lose* 100 points',
                     'Only if you respond fast, you earn the outcome of the choice (+100 or +0)',
                     '',
                     'If you see "ACC", you can take more time to decide',
                     'which makes it more likely that you are correct and earn points'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif self.session_location == 13:
                next_texts = self.generate_text_objects(
                    ['Let\'s see how this works!'],
                    bottom_pos=5, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

            elif self.session_location == 14:
                self.show_timing_feedback = True
                self.prepare_objects()  # re-create objects to ensure "outcome" and "reward" are separate
                # sorry about the ugliness of this quick-fix
                tr = AnnotatedTrial(trial_nr=trial_nr,
                                    parameters={'cue': 'ACC',
                                                'block_nr': self.session_location,
                                                'stimulus_symbol_left': self.all_symbols[4],
                                                'stimulus_symbol_right': self.all_symbols[5],
                                                'p_win_left': 1,
                                                'p_win_right': 1},
#                                    phase_durations=[5, 5, 0, 60, 0, 5, 0, 7, 0, 7, 60],
                                    phase_durations=[5, 5, 0, 60, 3, 0, 0, 7, 0, 7, 60],
                                    phase_names=self.phase_names + ['feedback_2', 'feedback_3'],
                                    # decoration_objects=next_texts + continue_back,
                                    session=self)
                tr.run()
                tr.last_key = tr.last_resp

            elif self.session_location == 15:
                self.show_timing_feedback = True
                self.prepare_objects()  # re-create objects to ensure "outcome" and "reward" are separate
                # sorry about the ugliness of this quick-fix

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

                for i in range(10):
                    symbols, ps = self.get_random_stimuli_ps()
                    symbols, ps, cue = self.get_random_cue_location(ps=ps,
                                                                    symbols=symbols)

                    tr = LearningTrial(trial_nr=trial_nr,
                                       parameters={'cue': cue,
                                                   'block_nr': self.session_location,
                                                   'stimulus_symbol_left': symbols[0],
                                                   'stimulus_symbol_right': symbols[1],
                                                   'p_win_left': ps[0],
                                                   'p_win_right': ps[1],
                                                   'n_trs': 5},
                                       phase_durations=self.get_jittered_durations(),
                                       phase_names=self.phase_names,
                                       session=self)
                    tr.run()
                    tr.last_key = tr.last_resp # for compatibility
                    trial_nr += 1

                self.move_forward()
                #session_location += 1  # manually forward here, since last key is not 'space'

            if self.session_location == 16:
                next_texts = self.generate_text_objects(
                    ['That\'s it! You\'re ready!\n\n'
                     'Please keep in mind the following:\n\n'
                     '- It is important to follow the speed/accuracy cues\n'
                     '- It can take some time to discover which symbols are correct, this is normal\n'
                     '- Press only one button per trial\n'
                     '- Keep your eyes on the fixation cross whenever it is present\n'
                     '- Sometimes, you see a black screen for about 10 seconds. This is normal\n'
                     '- It can be difficult, and that is normal. Stay focused and give it your best!'],
                    bottom_pos=-1, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()

                trial_nr += 1

            if self.session_location == 17:
                next_texts = self.generate_text_objects(
                    ['Next, you will again perform the symbol-learning task.', '',
                     'This time, there will be no cues.  Always respond as fast as possible',
                     'without making mistakes'],
                    bottom_pos=3, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()
                trial_nr += 1

            if self.session_location == 18:
                next_texts = self.generate_text_objects(
                    ['That\'s it! You\'re ready!\n\n'
                     'Please keep in mind the following:\n\n'
                     '- It can take some time to discover which symbols are correct, this is normal\n'
                     '- Press only one button per trial\n'
                     '- Keep your eyes on the fixation cross whenever it is present\n'
                     '- Sometimes, you see a black screen for about 10 seconds. This is normal\n'
                     '- It can be difficult, and that is normal. Stay focused and give it your best!'],
                    bottom_pos=-1, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()
                trial_nr += 1

            if self.session_location == 19:
                next_texts = self.generate_text_objects(
                    ['Next, you will again perform the symbol-learning task.',
                     'However, this time, there will be *cues*'],
                    bottom_pos=-1, degrees_per_line=1)
                tr = TextTrial(trial_nr=trial_nr,
                               parameters={},
                               phase_durations=[60],
                               decoration_objects=next_texts + continue_back,
                               session=self)
                tr.run()
                trial_nr += 1

            trial_nr += 1
            if tr.last_key == 'space':
                self.move_forward()
#                session_location += 1
            elif tr.last_key == 'backspace':
                self.move_backward()
#                session_location -= 1

            if self.stop:
                break

        self.close()
        if quit_on_exit:
            self.quit()


# if __name__ == '__main__':
#     import datetime
#
#     index_number = 1
#     start_block = 0
#     scanner = False
#     simulate = 'y'
#     debug = True
#
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
#     output_str = f'sub-{index_number}_task-practice-learning_datetime-{timestamp}'
#     output_dir = './data'
#     if simulate == 'y':
#         settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings_simulate.yml'
#     else:
#         settings_file = '/Users/steven/Sync/PhDprojects/IMCN-learningtask/IMCN-learningtask/settings.yml'
#
#     sess = PracticeSession(index_number=index_number,
#                            output_str=output_str,
#                            output_dir=output_dir,
#                            settings_file=settings_file,
#                            start_block=start_block)
#     sess.run()



'''
Session_locations, old version:

0: You will see two choice options
1: These act like slot machines in a casino
2: The probability of getting a reward differs per symbol
3: Practice single symbol 1
4: 'It\'s important to realise that:'
5: Let's practice again
6: 'Alright, well done so far!' 'The next aspect of the task are the *cues*'
7: 'Cues instruct you to ...'
8: 'Following the SPD / ACC cues is important!'
9: Let's see how this works
10: Slow example trial / "video"
11: Let's practice for real, 15 trials
12: 'Well done! There is one thing left to explain, differing symbols'
13: 'You will next practice a short block of trials with alternating symbols'
14: 'That\'s it! You\'re ready!'



New version:
0: Welcome
1: Important to read instructions carefully
2: You will see two choice options
3: These act like slot machines in a casino
4: The probability of getting a reward differs per symbol
5: Practice single symbol 1
6: 'It\'s important to realise that:'
7: Let's practice again

------
8: Introduce differing symbols
9: Practice 15 trials with differing symbols

------
10: 'Alright, well done so far!' 'The final aspect of the task are the *cues*'
11: 'Cues instruct you to ...'
12: 'Following the SPD / ACC cues is important!'
13:  Let's see how this works
14: Slow example trial / "video"
15: Let's practice for real, 15 trials
-------
16: You're ready SAT
----
17: Next, you will again perform the symbol-learning task. This time, there will be no cues. Always respond as fast 
as possible without making mistakes. Just as a recap, you will now practice 15 trials again.
18: You're ready Reversal

19: Next, you will again perform the symbol-learning task. However, it will be a bit more complex than before, 
since you will be presented with *cues*.


Presentation order:
If SAT first:
First practice: [0 - 16]
Second practice: [17, 9, 18]

If Reversal first:
First practice: [0-9, 19]
Second practice: [20, 11-16]


'''