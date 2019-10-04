from exptools2.core.trial import Trial
from psychopy import event, visual
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
        self.highlights_reset = False

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
        self.points_earned = 0
        self.current_feedback = [
            self.session.feedback_outcome_objects[2],   # no choice made
            self.session.feedback_earnings_objects[0],  # no points earned
        ]

        # Select cue, define deadline
        if parameters['cue'] == 'SPD':
            self.current_cue = self.session.cues[0]
            # TODO change this to a fixed deadline?
            # sample deadline for this trial from cumulative exponential distribution, parameterized by scale=1/2.5
            # and loc = 0.75.
            # note that the maximum deadline is always 2s
            self.deadline = 0.5   # np.min([np.random.exponential(scale=1/2.5)+.75, 2])
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

        # highlight correct stimulus before drawing the rectangle
        if self.phase == 3:
            self.stim[self.response['direction']].selected = True

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

        self.session.global_log.loc[idx, 'deadline'] = self.deadline
        self.session.global_log.loc[idx, 'rt'] = self.response['rt']
        self.session.global_log.loc[idx, 'rt_too_slow'] = self.response['too_slow']
        self.session.global_log.loc[idx, 'rt_too_fast'] = self.response['too_fast']
        self.session.global_log.loc[idx, 'rt_in_time'] = self.response['in_time']
        self.session.global_log.loc[idx, 'choice_key'] = self.response['choice_key']
        self.session.global_log.loc[idx, 'choice_direction'] = self.response['direction']
        self.session.global_log.loc[idx, 'choice_outcome'] = self.choice_outcome
        self.session.global_log.loc[idx, 'total_points_earned'] = self.session.total_points + self.points_earned

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

        if self.session.debug:
            self.update_debug_txt()
            self.session.debug_txt.draw()

        if self.phase == 0:  # Pre-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 1:  # Cue
            if self.current_cue is not None:
                self.current_cue.draw()

        elif self.phase == 2:  # Post-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 3:  # Stimulus
            if not self.highlights_reset:
                # make sure to reset so the highlighting rectangles from previous trial isn't drawn
                self.stim[0].selected = False
                self.stim[1].selected = False
                self.highlights_reset = True

            for stim in self.stim:
                stim.draw()
            self.session.fixation_cross.draw()

        elif self.phase == 4:  # Selection
            for stim in self.stim:
                stim.draw()
            self.session.fixation_cross.draw()

        elif self.phase == 5:  # feedback
            for fb in self.current_feedback:
                fb.draw()

            # if len(self.current_feedback) < 3:
            #     self.session.fixation_cross.draw()

        elif self.phase == 6:  # iti
            self.session.fixation_cross.draw()


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


class InstructionTrial(LearningTrial):

    def __init__(self, trial_nr, parameters, phase_durations,
                 decoration_objects=(), allow_space_break=False,
                 phase_names=None, session=None):
        super(InstructionTrial, self).__init__(trial_nr=trial_nr, parameters=parameters,
                                               phase_durations=phase_durations, phase_names=phase_names,
                                               session=session)
        self.allow_space_break = allow_space_break
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

                if self.allow_space_break:
                    if ev == 'space':
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

        for stim in self.decoration_objects:
            stim.draw()

        if self.phase == 5:
            # also draw during feedback
            for stim in self.stim:
                stim.draw()

        super(InstructionTrial, self).draw()


class CheckTrial(InstructionTrial):

    def __init__(self, trial_nr, parameters, phase_durations,
                 decoration_objects=(), allow_space_break=False,
                 phase_names=None, session=None):
        super(CheckTrial, self).__init__(trial_nr=trial_nr, parameters=parameters,
                                         decoration_objects=decoration_objects,
                                         allow_space_break=allow_space_break,
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
                                                    'press <->. Otherwise, press <space '
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

        if self.phase == 6:
            # also draw during feedback
            for stim in self.stim:
                stim.draw()

        super(CheckTrial, self).draw()


class PracticeTrial(LearningTrial):
    """ Practice trials do the same as normal trials, but annotate what happens in each different phase. """

    def __init__(self, trial_nr, parameters, phase_durations, annotate=True, decoration_objects=(),
                 phase_names=None, session=None):
        super(PracticeTrial, self).__init__(trial_nr, parameters, phase_durations,
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
                                           text='This indicates the trial is about to start. Please focus on this '
                                                'cross',
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
                    # visual.TextStim(win=self.session.win,
                    #                 text='Decide which you want to choose now!\nPress %s to choose left OR %s to '
                    #                      'choose right' %(self.session.response_button_signs[0],
                    #                                       self.session.response_button_signs[1]),
                    #                 pos=(0, -4), units='deg',
                    #                 wrapWidth=self.session.settings['text']['wrap_width']
                                    ],
                'phase_4': visual.TextStim(win=self.session.win,
                                           text='Your choice is highlighted shortly',
                                           pos=(0, 4), units='deg'),
                'phase_5': [
                    self.arrows[0],
                    visual.TextStim(win=self.session.win,
                                    text='This is the outcome of your choice',
                                    pos=(6, 0), units='deg',
                                    alignHoriz='left')],
                'phase_7': [
                    self.arrows[1],
                    visual.TextStim(win=self.session.win,
                                    text='This is your reward. You were in time, so your reward is '
                                         'the same as the outcome',
                                    pos=(8, 0), units='deg', alignHoriz='left')],
                'phase_8': [
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

        if self.phase == 7:  # Feedback is now split in three phases, for annotations of different components
            for fb in self.current_feedback:
                fb.draw()

        if self.phase == 8:
            self.current_feedback[0].draw()                   # outcome = whatever the participant got
            self.session.feedback_earnings_objects[2].draw()  # reward = 0
            self.session.feedback_timing_objects[0].draw()

        if self.phase in [0, 1, 2, 4, 5, 6, 7]:
            self.forward_text.draw()

        # Annotations
        if self.annotate:
            phase_n_str = 'phase_{}'.format(self.phase)
            if phase_n_str in self.annotations.keys():
                this_phase_annotations = self.annotations[phase_n_str]
                if not isinstance(this_phase_annotations, list):
                    this_phase_annotations = [this_phase_annotations]
                for annotation in this_phase_annotations:
                    annotation.draw()

        if np.isinf(self.phase_durations[self.phase]) and self.phase not in [4, 8, 9]:
            self.forward_text.draw()

        super(PracticeTrial, self).draw()

    def get_events(self):

        evs = event.getKeys(timeStamped=self.session.clock)

        for ev, time in evs:
            if len(ev) > 0:
                if ev == 'q':
                    self.session.close()
                    self.session.quit()
                elif ev == 'equal' or ev == '+':
                    self.stop_trial()

                if ev == 'space' and not self.phase == 3:
                    self.last_key = ev
                    self.stop_phase()

                if ev in ['space', 'backspace'] and self.phase == 8:
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
#
#     # def process_response(self, ev, time):
#     #
#     #     super(PracticeTrial, self).process_response(ev, time)
#     #     if len(self.current_feedback)
#
#     def run(self):
#         """
#         Runs this trial
#         """
#
#         self.start_time = self.session.clock.getTime()
#         if self.tracker:
#             self.tracker.log('trial ' + str(self.trial_nr) + ' started at ' + str(self.start_time))
#             self.tracker.send_command('record_status_message "Trial ' + str(self.trial_nr) + '"')
#         self.events.append('trial ' + str(self.trial_nr) + ' started at ' + str(self.start_time))
#
#         while not self.stopped:
#             self.run_time = self.session.clock.getTime() - self.start_time
#
#             # Waits for scanner pulse.
#             if self.phase == 0:
#                 self.t_time = self.session.clock.getTime()
#                 if not self.session.in_scanner:
#                     self.phase_forward()
#
#             # In phase 1, we show fix cross (jittered timing)
#             if self.phase == 1:
#                 self.jitter_time_1 = self.session.clock.getTime()
#                 if (self.jitter_time_1 - self.t_time) > self.phase_durations[1]:
#                     self.phase_forward()
#
#             # In phase 2, we show the cue (fixed timing)
#             if self.phase == 2:
#                 self.cue_time = self.session.clock.getTime()
#                 if (self.cue_time - self.jitter_time_1) > self.phase_durations[2]:
#                     self.phase_forward()
#
#             # In phase 3, we show the fixation cross again (jittered timing)
#             if self.phase == 3:
#                 self.jitter_time_2 = self.session.clock.getTime()
#                 if (self.jitter_time_2 - self.cue_time) > self.phase_durations[3]:
#                     self.phase_forward()
#
#             # In phase 4, we show the stimulus
#             if self.phase == 4:
#                 self.stimulus_time = self.session.clock.getTime()
#                 if (self.stimulus_time - self.jitter_time_2) > self.phase_durations[4]:
#                     self.phase_forward()
#
#             # In phase 5, we highlight the chosen option
#             if self.phase == 5:
#                 self.selection_time = self.session.clock.getTime()
#                 if (self.selection_time - self.stimulus_time) > self.phase_durations[5]:
#                     self.phase_forward()
#
#             # In phase 6, we show feedback
#             if self.phase == 6:
#                 self.feedback_time_1 = self.session.clock.getTime()
#                 if (self.feedback_time_1 - self.selection_time) > self.phase_durations[6]:
#                     self.phase_forward()
#
#             # In phase 7, we show feedback
#             if self.phase == 7:
#                 self.feedback_time_2 = self.session.clock.getTime()
#                 if (self.feedback_time_2 - self.feedback_time_1) > self.phase_durations[7]:
#                     self.phase_forward()
#
#             # In phase 8, we show feedback
#             if self.phase == 8:
#                 self.feedback_time_3 = self.session.clock.getTime()
#                 if (self.feedback_time_3 - self.feedback_time_2) > self.phase_durations[8]:
#                     self.phase_forward()
#
#             # ITI
#             if self.phase == 9:
#                 self.iti_time = self.session.clock.getTime()
#                 if (self.iti_time - self.feedback_time_3) > self.phase_durations[9]:
#                     self.phase_forward()
#                     self.stopped = True
#
#             # events and draw, but only if we haven't stopped yet
#             if not self.stopped:
#                 self.event()
#                 self.draw()
#
#                 # # make screen shots
#                 # ss_fn = 'screenshots/trial_%d_phase_%d_set_%d.png' % (self.ID, self.phase, self.parameters[
#                 #     'stimulus_set'])
#                 # if not os.path.isfile(ss_fn):
#                 #     self.screen.getMovieFrame()  # Defaults to front buffer, I.e. what's on screen now.
#                 #     self.screen.saveMovieFrames(ss_fn)
#
#         self.stop()
