from exptools2.core.trial import Trial
from psychopy import event, visual, core
import numpy as np


class LearningTrial(Trial):

    def __init__(self, trial_nr, parameters, phase_durations, phase_names=None,
                 session=None):
        super(LearningTrial, self).__init__(trial_nr=trial_nr,
                                            parameters=parameters,
                                            phase_durations=phase_durations,
                                            phase_names=phase_names,
                                            session=session)
        self.trial_nr = trial_nr
        self.parameters = parameters
        self.choice_outcome = None
        self.trs_recorded = 0
        if 'n_trs' in parameters.keys():
            self.stop_on_tr = parameters['n_trs']
        else:
            self.stop_on_tr = None

        # set-up response options / dict
        self.response_measured = False  # Has the pp responded yet?
        self.response = {}
        self.response['choice_key'] = None
        self.response['rt'] = None
        self.response['direction'] = None
        self.response['too_slow'] = None
        self.response['too_fast'] = None
        self.response['in_time'] = None

        # Select stimulus
        self.stim = [self.session.stimuli['left'][parameters['stimulus_symbol_left']],
                     self.session.stimuli['right'][parameters['stimulus_symbol_right']]]

        # Select feedback
        self.points_earned = -100  # assuming no response. Will be overwritten with a response
        self.current_feedback = [
            self.session.feedback_outcome_objects[2],   # no choice made
            self.session.feedback_earnings_objects[2],  # -100 points
        ]

        # Select cue, define deadline
        if parameters['cue'] == 'SPD':
            self.current_cue = self.session.cues[0]
            # TODO change this to a fixed deadline?
            # sample deadline for this trial from cumulative exponential distribution, parameterized by scale=1/2.5
            # and loc = 0.75.
            # note that the maximum deadline is always 2s
            self.deadline = 0.70   # np.min([np.random.exponential(scale=1/2.5)+.75, 2])
        elif parameters['cue'] == 'ACC':
            self.current_cue = self.session.cues[1]
            self.deadline = self.phase_durations[3]
        else:
            self.current_cue = None
            self.deadline = self.phase_durations[3]

    def get_events(self):
        """ evs, times can but used to let a child object pass on the evs and times """

        evs = event.getKeys(timeStamped=self.session.clock)

        for ev, time in evs:
            if len(ev) > 0:
                if ev == 'q':
                    self.session.close()
                    self.session.quit()
                elif ev == 'equal' or ev == '+':
                    self.stop_trial()

                idx = self.session.global_log.shape[0]

                # TR pulse
                if ev == self.session.mri_trigger:
                    event_type = 'pulse'
                    self.trs_recorded += 1

                    if self.stop_on_tr is not None:
                        # Trial ends whenever trs_recorded >= preset number of trs
                        if self.trs_recorded >= self.stop_on_tr:
                            self.stop_trial()

                elif ev in self.session.response_button_signs:
                    event_type = 'response'
                    if self.phase == 3 or self.phase == 4:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time, idx)

                            # Not in the MR scanner? End phase upon keypress
                            # if not self.session.in_scanner and self.phase == 3:
                            #     self.stop_phase()
                else:
                    event_type = 'non_response_keypress'

                # global response handling
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'block_nr'] = self.parameters['block_nr']
                self.session.global_log.loc[idx, 'onset'] = time
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{ev}_time-{time}'
                    self.session.tracker.sendMessage(msg)

                if ev != self.session.mri_trigger:
                    self.last_resp = ev
                    self.last_resp_onset = time

    def make_feedback_screen(self):
        # set-up feedback: 1. Did the pp win?
        self.current_feedback = []
        if self.choice_outcome:
            self.current_feedback.append(self.session.feedback_outcome_objects[1])
        elif not self.choice_outcome:
            self.current_feedback.append(self.session.feedback_outcome_objects[0])

        # 2. How many points were actually earned?
        if self.response['too_fast']:
            self.current_feedback.append(self.session.feedback_earnings_objects[0])
        elif self.response['too_slow']:
            self.current_feedback.append(self.session.feedback_earnings_objects[2])
        else:
            if self.choice_outcome:
                self.current_feedback.append(self.session.feedback_earnings_objects[1])
            else:
                self.current_feedback.append(self.session.feedback_earnings_objects[0])

        # 3. Too slow / too fast?
        if self.response['too_slow']:
            self.current_feedback.append(self.session.feedback_timing_objects[0])
        if self.response['too_fast']:
            self.current_feedback.append(self.session.feedback_timing_objects[1])

    def process_response(self, ev, time, idx):
        """ Processes a response:
        - checks if the keypress is correct/incorrect;
        - checks if the keypress was in time;
        - Prepares feedback accordingly """

        self.response['choice_key'] = ev

        # to calculate response time, look up stimulus onset time
        log = self.session.global_log
        stim_onset_time = log.loc[(log.trial_nr == self.trial_nr) & (log.event_type == 'stimulus'), 'onset'].values[0]

        self.response['rt'] = time - stim_onset_time
        self.response['direction'] = np.where(np.array(self.session.response_button_signs) == ev)[0][0]
        self.response['too_slow'] = self.response['rt'] >= self.deadline
        self.response['too_fast'] = self.response['rt'] <= .1
        self.response['in_time'] = 0.1 < self.response['rt'] < self.deadline

        # highlight chosen stimulus before drawing the rectangle
        # if self.phase_names[self.phase] == 'stimulus':
        #     self.stim[self.response['direction']].selected = True

        # What is the outcome of the choice?
        if self.response['direction'] == 0:
            p_choice = self.parameters['p_win_left']
        else:
            p_choice = self.parameters['p_win_right']

        self.choice_outcome = np.random.binomial(1, p_choice, 1)[0]  # if participant won, this is 1

        # if the response was in time, the participant gains the reward
        if self.response['in_time']:
            self.points_earned = self.choice_outcome * 100   # 100 points per win
        elif self.response['too_slow']:
            self.points_earned = -100

        self.make_feedback_screen()

        # self.session.global_log.loc[idx, 'deadline'] = self.deadline
        self.session.global_log.loc[idx, 'rt'] = self.response['rt']
        # self.session.global_log.loc[idx, 'rt_too_slow'] = self.response['too_slow']
        # self.session.global_log.loc[idx, 'rt_too_fast'] = self.response['too_fast']
        # self.session.global_log.loc[idx, 'rt_in_time'] = self.response['in_time']
        # self.session.global_log.loc[idx, 'choice_key'] = self.response['choice_key']
        self.session.global_log.loc[idx, 'choice_direction'] = self.response['direction']
        self.session.global_log.loc[idx, 'choice_outcome'] = self.choice_outcome
#        self.session.global_log.loc[idx, 'total_points_earned'] = self.session.total_points + self.points_earned

    def update_debug_txt(self):

        trial_nr = self.trial_nr
        phase = self.phase
        time = self.session.clock.getTime()
        trial_start_time = self.start_trial
        if trial_start_time is None:
            trial_start_time = 0
        t_start = time - trial_start_time
        stim = [self.parameters['stimulus_symbol_left'], self.parameters['stimulus_symbol_right']]
        ps = [self.parameters['p_win_left'], self.parameters['p_win_right']]

        debug_str = 'Trial: %d\nPhase: %d\nTime since start: %.3fs\nTrial start time: %.3fs\nTime since trial start: %.3fs\nStim set: %s\nPs: [%.1f, %.1f]\nTotal points earned: %d\nDeadline: %.2fs' %(
        trial_nr, phase, time, trial_start_time, t_start, stim, ps[0], ps[1], self.session.total_points, self.deadline)
        resp_str = "\n".join(("{}: {}".format(*i) for i in self.response.items()))
        debug_str = debug_str + '\n' + resp_str
        self.session.debug_txt.text = debug_str

    def draw(self):
        """
        Phases:
        0 = fixation cross 1
        1 = cue
        2 = fixation cross 2
        3 = stimulus
        4 = fixation cross 3
        5 = highlight
        6 = fixation cross 4
        7 = feedback
        8 = fixation cross 5 (iti)
        """

        if self.session.debug:
            self.update_debug_txt()
            self.session.debug_txt.draw()

        if self.phase in [0, 2, 3, 4, 5, 6, 8]:
            self.session.fixation_cross.draw()

        if self.phase == 1:  # Cue
            if self.current_cue is not None:
                self.current_cue.draw()

        if self.phase == 3 or self.phase == 5:  # Draw Stimulus

            for stim in self.stim:
                stim.draw()

            # # Sim
            # if not self.response_measured and self.session.timer.getTime() >= -1:
            #     self.response_measured = True
            #     time = self.session.clock.getTime()
            #     self.process_response('z', time, self.session.global_log.shape[0])

        if self.phase == 5:  # Selection
            if self.response['direction'] is not None:
                self.session.selection_rectangles[self.response['direction']].draw()

        if self.phase == 7:  # feedback
            for fb in self.current_feedback:
                fb.draw()


class TextTrial(Trial):

    def __init__(self, trial_nr, parameters, phase_durations, #text_stim_params,
                 decoration_objects = (),
                 phase_names=None, session=None):
        super(TextTrial, self).__init__(trial_nr=trial_nr, parameters=parameters,
                                        phase_durations=phase_durations, phase_names=phase_names,
                                        session=session)

        self.decoration_objects = decoration_objects
        # self.text_stims = []
        # for param_set in text_stim_params:
        #     self.text_stims.append(visual.TextStim(win=self.session.win, **param_set))

        self.last_key = None

    def draw(self):

        # for text_stim in self.text_stims:
        #     text_stim.draw()
        for stim in self.decoration_objects:
            stim.draw()

    def get_events(self):
        evs = event.getKeys(timeStamped=self.session.clock)

        for ev, time in evs:
            if len(ev) > 0:
                if ev == 'q':
                    self.session.close()
                    self.session.quit()
                elif ev == 'equal' or ev == '+' or ev == 'space' or ev == '-' or ev == 'minus' or ev == 'backspace':
                    self.last_key = ev
                    self.stop_trial()


class EndOfBlockTrial(TextTrial):

    def __init__(self, trial_nr, parameters, phase_durations, bottom_pos=0, degrees_per_line=1, wrapWidth=100,
                 phase_names=None, session=None, **kwargs):
        super(EndOfBlockTrial, self).__init__(trial_nr=trial_nr, parameters=parameters, decoration_objects=(),
                                              phase_durations=phase_durations, phase_names=phase_names,
                                              session=session)

        pts = self.session.total_points
        if pts > 0:
            txt = 'You earned {} points so far. Well done!'.format(pts)
        else:
            txt = 'You earned {} points so far'.format(pts)
        text_per_line = ['End of this block',
                         txt]
        text_objects = []
        for i, text in enumerate(text_per_line):
            text_objects.append(visual.TextStim(self.session.win, text=text,
                                                pos=(0, bottom_pos-i*degrees_per_line),
                                                alignVert='bottom', wrapWidth=wrapWidth, **kwargs))
        text_objects.append(visual.TextStim(self.session.win, text='Waiting for operator...', italic=True,
                                            pos=(0, -5)))
        self.decoration_objects = text_objects


class InstructionTrial(LearningTrial):

    def __init__(self, trial_nr, parameters, phase_durations,
                 decoration_objects=(), break_keys='space',
                 phase_names=None, session=None):
        super(InstructionTrial, self).__init__(trial_nr=trial_nr, parameters=parameters,
                                               phase_durations=phase_durations, phase_names=phase_names,
                                               session=session)

        if break_keys is None:
            break_keys = []
        elif isinstance(break_keys, str):
            break_keys = [break_keys]

        self.break_keys = break_keys
        self.decoration_objects = decoration_objects
        # self.instruction_txt = [visual.TextStim(self.session.win, text='test',
        #                                         pos=(0, 4))]
        self.deadline = 60
        self.last_key = None

    def make_feedback_screen(self):

        # set-up feedback: 1. Did the pp win? only
        self.current_feedback = []
        if self.choice_outcome:
            self.current_feedback.append(self.session.feedback_outcome_objects[1])
        elif not self.choice_outcome:
            self.current_feedback.append(self.session.feedback_outcome_objects[0])

    def get_events(self):
        """ evs, times can but used to let a child object pass on the evs and times """

        evs = event.getKeys(timeStamped=self.session.clock)

        for ev, time in evs:
            if len(ev) > 0:
                if ev == 'q':
                    self.session.close()
                    self.session.quit()
                elif ev == 'equal' or ev == '+':
                    self.stop_trial()

                if ev in self.break_keys:
                    self.last_key = ev
                    self.stop_trial()

                idx = self.session.global_log.shape[0]

                # TR pulse
                if ev == self.session.mri_trigger:
                    event_type = 'pulse'
                elif ev in self.session.response_button_signs:
                    event_type = 'response'
                    if self.phase == 3 or self.phase == 4:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time, idx)

                            # Not in the MR scanner? End phase upon keypress
                            if not self.session.in_scanner and self.phase == 3:
                                self.stop_phase()
                else:
                    event_type = 'non_response_keypress'

                # global response handling
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'block_nr'] = self.parameters['block_nr']
                self.session.global_log.loc[idx, 'onset'] = time
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{ev}_time-{time}'
                    self.session.tracker.sendMessage(msg)

                if ev != self.session.mri_trigger:
                    self.last_resp = ev
                    self.last_resp_onset = time

    def draw(self):

        if self.phase == 7:
            # # also draw during feedback
            for stim in self.decoration_objects[:-1]:
                stim.draw()
        else:
            for stim in self.decoration_objects:
                stim.draw()

        super(InstructionTrial, self).draw()


class CheckTrial(InstructionTrial):

    def __init__(self, trial_nr, parameters, phase_durations,
                 decoration_objects=(), break_keys=('space', 'backspace'),
                 phase_names=None, session=None):
        super(CheckTrial, self).__init__(trial_nr=trial_nr, parameters=parameters,
                                         decoration_objects=decoration_objects, break_keys=break_keys,
                                         phase_durations=phase_durations, phase_names=phase_names,
                                         session=session)

        self.correct_feedback = [visual.TextStim(win=self.session.win,
                                                 text='Correct!'),
                                 visual.TextStim(win=self.session.win,
                                                 text='Press <space bar> to continue',
                                                 pos=(0, -6),
                                                 italic=True,
                                                 wrapWidth=100)]

        self.error_feedback = [visual.TextStim(win=self.session.win,
                                               text='Wrong!'),
                               visual.TextStim(win=self.session.win,
                                               text='If you want to do some more trials for this stimulus, '
                                                    'press <backspace>. Otherwise, press <space '
                                                    'bar> to continue',
                                               pos=(0, -6),
                                               italic=True,
                                               wrapWidth=100
                                               )]

    def make_feedback_screen(self):
        if self.response['direction'] == 0:
            self.current_feedback = self.correct_feedback
        else:
            self.current_feedback = self.error_feedback

    def draw(self):

        if self.phase == 7:
            # also draw during feedback
            for stim in self.stim:
                stim.draw()
            self.session.selection_rectangles[self.response['direction']].draw()

        super(CheckTrial, self).draw()


class AnnotatedTrial(LearningTrial):
    """ Does the same as normal trials, but annotate what happens in each different phase. """

    def __init__(self, trial_nr, parameters, phase_durations, annotate=True, decoration_objects=(),
                 phase_names=None, session=None):
        super(AnnotatedTrial, self).__init__(trial_nr, parameters, phase_durations,
                                            phase_names=phase_names, session=session)

        # prepare arrow
        arrow_left_vertices = [(0.2, 0.05),
                               (0.2, -0.05),
                               (0.0, -0.05),
                               (0, -0.1),
                               (-0.2, 0),
                               (0, 0.1),
                               (0, 0.05)]

        arrow_y_pos = self.session.settings['text']['feedback_y_pos']

        self.decoration_objects = decoration_objects

        self.arrows = [visual.ShapeStim(win=self.session.win,
                                        vertices=[(x*2+4, y*2+arrow_y_pos[0]) for x, y in arrow_left_vertices],
                                        fillColor='lightgray', size=1, lineColor='lightgray', units='deg'),

                       visual.ShapeStim(win=self.session.win,
                                        vertices=[(x*2+4, y*2+arrow_y_pos[1]) for x, y in arrow_left_vertices],
                                        fillColor='lightgray', size=1, lineColor='lightgray', units='deg')]

        self.annotate = annotate
        if self.annotate:
            if parameters['cue'] == 'SPD':
                spdacc = 'fast'
            else:
                spdacc = 'accurate'
            self.forward_text = visual.TextStim(win=self.session.win,
                                                text='Press <space bar> to continue',
                                                italic=True,
                                                pos=(0, -4), units='deg',
                                                wrapWidth=self.session.settings['text']['wrap_width'],
                                                alignHoriz='center'
                                                )
            self.annotations = {
                'phase_0': visual.TextStim(win=self.session.win,
                                           text='This indicates the trial is about to start. Please focus your eyes on '
                                                'this cross',
                                           pos=(0, 4), units='deg',
                                           wrapWidth=self.session.settings['text']['wrap_width']
                                           ),
                'phase_1': visual.TextStim(win=self.session.win,
                                           text='This cue tells you to be fast or accurate. In the upcoming choice, '
                                                'you need to be %s!' % spdacc,
                                           pos=(0, 4), units='deg',
                                           wrapWidth=self.session.settings['text']['wrap_width']
                                           ),
                'phase_3': [
                    visual.TextStim(win=self.session.win,
                                    text='These are your choice options. Make your choice now',
                                    pos=(0, 4), units='deg')
                                    ],
                'phase_3.5': [
                    visual.TextStim(win=self.session.win,
                                    text='Your choice is recorded, the trial will continue soon...',
                                    pos=(0, 4), units='deg')
                                    ],
                'phase_4': [
                    visual.TextStim(win=self.session.win,
                                    text='The symbols disappear',
                                    pos=(0, 4), units='deg')
                                    ],
                'phase_5': visual.TextStim(win=self.session.win,
                                           text='Your choice is highlighted',
                                           pos=(0, 4), units='deg'),
                'phase_6': [
                    visual.TextStim(win=self.session.win,
                                    text='The highlight disappears',
                                    pos=(0, 4), units='deg')
                                    ],
                'phase_7': [
                    self.arrows[0],
                    visual.TextStim(win=self.session.win,
                                    text='This is the outcome of your choice',
                                    pos=(8, 0), units='deg',
                                    alignHoriz='left')],
                'phase_9': [
                    self.arrows[1],
                    visual.TextStim(win=self.session.win,
                                    text='This is your reward. You were in time, so your reward is '
                                         'the same as the outcome',
                                    pos=(8, 0), units='deg', alignHoriz='left')],
                'phase_10': [
                    visual.TextStim(win=self.session.win,
                                    text='If you are too late, you *lose* 100 points '
                                         'whether the outcome was +0 or +100',
                                    pos=(8, 0), units='deg', alignHoriz='left',
                                    ),
                    visual.TextStim(win=self.session.win,
                                    text='Press <backspace> to restart this explanation',
                                    pos=(0, -4), units='deg', alignVert='top',
                                    wrapWidth=self.session.settings['text']['wrap_width']
                                    ),
                    visual.TextStim(win=self.session.win,
                                    text='Press <space bar> to continue',
                                    pos=(0, -5), units='deg', alignVert='top',
                                    wrapWidth=self.session.settings['text']['wrap_width']
                                    )
                ]
            }

    def draw(self):

        for stim in self.decoration_objects:
            stim.draw()

        if self.phase == 9:  # Feedback is now split in three phases, for annotations of different components
            for fb in self.current_feedback:
                fb.draw()

        if self.phase == 10:
            self.current_feedback[0].draw()                   # outcome = whatever the participant got
            self.session.feedback_earnings_objects[2].draw()  # reward = 0
            self.session.feedback_timing_objects[0].draw()

        # if self.phase in [9]:  # [0, 1, 2, 4, 5, 6, 7, 9]:
        #     self.forward_text.draw()

        # Annotations
        if self.annotate:
            phase_n_str = 'phase_{}'.format(self.phase)
            if phase_n_str in self.annotations.keys():
                this_phase_annotations = self.annotations[phase_n_str]
                if not isinstance(this_phase_annotations, list):
                    this_phase_annotations = [this_phase_annotations]

                if self.phase == 3 and self.response_measured:
                    # overwrite if response is recorded
                    this_phase_annotations = self.annotations['phase_3.5']

                for annotation in this_phase_annotations:
                    annotation.draw()

        super(AnnotatedTrial, self).draw()

    def get_events(self):

        evs = event.getKeys(timeStamped=self.session.clock)

        for ev, time in evs:
            if len(ev) > 0:
                if ev == 'q':
                    self.session.close()
                    self.session.quit()
                elif ev == 'equal' or ev == '+':
                    self.stop_trial()
                elif ev == 'minus':
                    self.stop_phase()  #super secret key

                if ev == 'space' and self.phase == 8:
                    self.last_key = ev
                    self.stop_phase()

                if ev in ['space', 'backspace'] and self.phase in [10]:
                    self.last_key = ev
                    self.stop_phase()

                idx = self.session.global_log.shape[0]

                # TR pulse
                if ev == self.session.mri_trigger:
                    event_type = 'pulse'
                elif ev in self.session.response_button_signs:
                    event_type = 'response'
                    if self.phase == 3 or self.phase == 4:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time, idx)

                            # reset timer, add 4 seconds
                            self.session.timer.reset()
                            self.session.timer.add(4)

                else:
                    event_type = 'non_response_keypress'

                # global response handling
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'block_nr'] = self.parameters['block_nr']
                self.session.global_log.loc[idx, 'onset'] = time
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{ev}_time-{time}'
                    self.session.tracker.sendMessage(msg)

                if ev != self.session.mri_trigger:
                    self.last_resp = ev
                    self.last_resp_onset = time
