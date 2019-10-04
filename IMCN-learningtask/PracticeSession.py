from LearningTrial import LearningTrial, InstructionTrial, TextTrial, CheckTrial, PracticeTrial
from psychopy import visual
from psychopy.visual import TextStim
from LearningSession import LearningSession
import numpy as np
import pandas as pd


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
        """ Generates some simple, jittered durations """

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
        session_location = self.start_block
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
