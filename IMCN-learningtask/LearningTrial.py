from exptools.core.trial import MRITrial
from exptools.core.session import MRISession
from psychopy import event, visual
import numpy as np
import os


class LearningTrial(MRITrial):

    def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
        super(LearningTrial, self).__init__(parameters=parameters,
                                            phase_durations=phase_durations,
                                            session=session,
                                            screen=screen,
                                            tracker=tracker)

        self.trs_recorded = 0
        self.ID = ID
        self.parameters = parameters
        self.won = None

        # set-up response options / dict
        self.response_measured = False  # Has the pp responded yet?
        self.response = {}
        self.response['button'] = None
        self.response['rt'] = None
        self.response['side'] = None
        # self.response['accuracy'] = None
        self.response['too_slow'] = None
        self.response['too_fast'] = None
        self.response['in_time'] = None

        # Select stimulus
        self.stim = [self.session.stimuli['left'][parameters['stim_left']],
                     self.session.stimuli['right'][parameters['stim_right']]]

        # self.stim = self.session.stimuli[parameters['stimulus_set']]
        self.stim[0].selected = False  # make sure to reset so the highlighting rectangle from previous trial doesnt
        # get drawn
        self.stim[1].selected = False  # make sure to reset

        # Select feedback
        self.points_earned = 0
        self.current_feedback = [
            self.session.feedback_outcome_objects[2],  # no choice made
            self.session.feedback_earnings_objects[0],  # no points earned
        ]
#        self.current_feedback = self.session.feedback_text_objects[0]  # default feedback (no response) is 'too slow'

        # Select cue, define deadline
        if parameters['cue'] == 'SPD':
            self.current_cue = self.session.cues[0]
            # sample deadline for this trial from cumulative exponential distribution, parameterized by scale=1/2.5
            # and loc = 0.75.
            # note that the maximum deadline is always 2s
            self.deadline = np.min([np.random.exponential(scale=1/2.5)+.75, 2])
        elif parameters['cue'] == 'ACC':
            self.current_cue = self.session.cues[1]
            self.deadline = self.phase_durations[4]
        else:
            self.current_cue = None
            self.deadline = self.phase_durations[4]

        # initialize times
        self.t_time = self.jitter_time_1 = self.cue_time = self.jitter_time_2 = self.stimulus_time = \
            self.selection_time = self.feedback_time = self.iti_time = -1

    def event(self, ev=None, time=None):
        """ evs, times can but used to let a child object pass on the evs and times """

        if ev is None:
            # no ev passed by child, get events here
            evs = event.getKeys(timeStamped=self.session.clock)
        else:
            evs = [(ev, time)]

        for ev, time in evs:   # event.getKeys(timeStamped=self.session.clock):
            if len(ev) > 0:

                # Quit experiment?
                if ev in ['esc', 'escape']:
                    self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                    print('run canceled by user')

                # Skip trial
                elif ev == '+' or ev == 'equal':
                    self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    print('trial canceled by user')

                # TR pulse
                elif ev == self.session.mri_trigger_key:
                    self.trs_recorded += 1
                    if self.phase == 0:
                        if self.trs_recorded >= self.session.warmup_trs:
                            if self.parameters['block_trial_ID'] == 0:
                                # make sure to register this pulse now as the start of the block/run
                                self.session.block_start_time = time

                    self.events.append([99,
                                        time,  # absolute time since session start
                                        time - self.start_time,  # time since trial start
                                        time - self.session.block_start_time,  # time since block start
                                        self.ID])  # trial ID seems useful maybe

                    # phase 0 is ended by the MR trigger
                    if self.phase == 0:
                        if self.parameters['block_trial_ID'] == 0:
                            if self.trs_recorded >= self.session.warmup_trs:
                                self.phase_forward()
                        else:
                            self.phase_forward()

                # Response (key press)
                elif ev in self.session.response_button_signs:
                    self.events.append([ev, time,  # absolute time since start of experiment
                                        self.session.clock.getTime() - self.start_time,  # time since start of trial
                                        self.stimulus_time - self.jitter_time_2,  # time since stimulus start
                                        'key_press'])

                    if self.phase == 4 or self.phase == 5:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time)

                            # Not in the MR scanner? End phase upon keypress
                            if self.session.tr == 0 and self.phase == 4:
                                self.phase_forward()

            super(LearningTrial, self).key_event(ev)

    def process_response(self, ev, time):
        """ Processes a response:
        - checks if the keypress is correct/incorrect;
        - checks if the keypress was in time;
        - Prepares feedback accordingly """

        self.response['button'] = ev
        self.response['rt'] = time - self.jitter_time_2
        self.response['side'] = np.where(np.array(self.session.response_button_signs) == ev)[0][0]
#        self.response['accuracy'] = ev == self.parameters['correct_response']
        self.response['too_slow'] = self.response['rt'] >= self.deadline
        self.response['too_fast'] = self.response['rt'] <= .1
        self.response['in_time'] = 0.1 < self.response['rt'] < self.deadline

        # highlight correct stimulus before drawing the rectangle
        self.stim[self.response['side']].selected = True
        # self.stim.selected = self.response['side']

        # What is the outcome of the choice?
        p_choice = self.parameters['p_win'][self.response['side']]  # get probability of winning of current choice
        self.won = np.random.binomial(1, p_choice, 1)[0]  # did the participant win?

        # if the response was in time, the participant gains the reward
        if self.response['in_time']:
            self.points_earned = self.won * 100  # 100 points per win

        # set-up feedback: 1. Did the pp win?
        self.current_feedback = []
        if self.won:
            self.current_feedback.append(self.session.feedback_outcome_objects[1])
        elif not self.won:
            self.current_feedback.append(self.session.feedback_outcome_objects[0])

        # 2. How many points were actually earned?
        if self.response['too_slow'] or self.response['too_fast']:
            self.current_feedback.append(self.session.feedback_earnings_objects[0])
        else:
            if self.won:
                self.current_feedback.append(self.session.feedback_earnings_objects[1])
            else:
                self.current_feedback.append(self.session.feedback_earnings_objects[0])

        # 3. Too slow / too fast?
        if self.response['too_slow']:
            self.current_feedback.append(self.session.feedback_timing_objects[0])
        if self.response['too_fast']:
            self.current_feedback.append(self.session.feedback_timing_objects[1])

    def run(self):
        """
        Runs this trial
        """

        self.start_time = self.session.clock.getTime()
        if self.tracker:
            self.tracker.log('trial ' + str(self.ID) + ' started at ' + str(self.start_time))
            self.tracker.send_command('record_status_message "Trial ' + str(self.ID) + '"')
        self.events.append('trial ' + str(self.ID) + ' started at ' + str(self.start_time))

        while not self.stopped:
            self.run_time = self.session.clock.getTime() - self.start_time

            # Waits for scanner pulse.
            if self.phase == 0:
                self.t_time = self.session.clock.getTime()
                if not isinstance(self.session, MRISession) or self.session.tr == 0:
                    self.phase_forward()

            # In phase 1, we show fix cross (jittered timing)
            if self.phase == 1:
                self.jitter_time_1 = self.session.clock.getTime()
                if (self.jitter_time_1 - self.t_time) > self.phase_durations[1]:
                    self.phase_forward()

            # In phase 2, we show the cue (fixed timing)
            if self.phase == 2:
                self.cue_time = self.session.clock.getTime()
                if (self.cue_time - self.jitter_time_1) > self.phase_durations[2]:
                    self.phase_forward()

            # In phase 3, we show the fixation cross again (jittered timing)
            if self.phase == 3:
                self.jitter_time_2 = self.session.clock.getTime()
                if (self.jitter_time_2 - self.cue_time) > self.phase_durations[3]:
                    self.phase_forward()

            # In phase 4, we show the stimulus
            if self.phase == 4:
                self.stimulus_time = self.session.clock.getTime()
                if (self.stimulus_time - self.jitter_time_2) > self.phase_durations[4]:
                    self.phase_forward()

            # In phase 5, we highlight the chosen option
            if self.phase == 5:
                self.selection_time = self.session.clock.getTime()
                if (self.selection_time - self.stimulus_time) > self.phase_durations[5]:
                    self.phase_forward()

            # In phase 6, we show feedback
            if self.phase == 6:
                self.feedback_time = self.session.clock.getTime()
                if (self.feedback_time - self.selection_time) > self.phase_durations[6]:
                    self.phase_forward()

            # ITI
            if self.phase == 7:
                self.iti_time = self.session.clock.getTime()
                if (self.iti_time - self.feedback_time) > self.phase_durations[7]:
                    self.phase_forward()
                    self.stopped = True

            # events and draw, but only if we haven't stopped yet
            if not self.stopped:
                self.event()
                self.draw()

                # # make screen shots
                # ss_fn = 'screenshots/trial_%d_phase_%d_set_%d.png' %(self.ID, self.phase, self.parameters[
                #     'stimulus_set'])
                # if not os.path.isfile(ss_fn):
                #     self.screen.getMovieFrame()   # Defaults to front buffer, I.e. what's on screen now.
                #     self.screen.saveMovieFrames(ss_fn)

        self.stop()

    def update_debug_txt(self):

        trial = self.ID
        phase = self.phase
        time = self.session.clock.getTime()
        trial_start_time = self.start_time
        t_start = time - trial_start_time
        stim = [self.parameters['stim_left'], self.parameters['stim_right']]
        ps = self.parameters['p_win']

        debug_str = 'Trial: %d\nPhase: %d\nTime since start: %.3fs\nTrial start time: %.3fs\nTime since trial start: %.3fs\nStim set: %s\nPs: [%.1f, %.1f]\nTotal points earned: %d\nDeadline: %.2fs' %(
        trial, phase, time, trial_start_time, t_start, stim, ps[0], ps[1], self.session.total_points, self.deadline)
        resp_str = "\n".join(("{}: {}".format(*i) for i in self.response.items()))
        debug_str = debug_str + '\n' + resp_str
        self.session.debug_txt.text = debug_str

    def draw(self):

        if self.session.debug:
            self.update_debug_txt()
            self.session.debug_txt.draw()

        if self.phase == 0:   # waiting for scanner-time
            if self.parameters['block_trial_ID'] == 0:
                self.session.scanner_wait_screen.draw()
            else:
                self.session.fixation_cross.draw()

        elif self.phase == 1:  # Pre-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 2:  # Cue
            if self.current_cue is not None:
                self.current_cue.draw()

        elif self.phase == 3:  # Post-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 4:  # Stimulus
            for stim in self.stim:
                stim.draw()
            self.session.fixation_cross.draw()

        elif self.phase == 5:  # Selection
            for stim in self.stim:
                stim.draw()
            self.session.fixation_cross.draw()

        elif self.phase == 6:  # feedback
            for fb in self.current_feedback:
                fb.draw()

            if len(self.current_feedback) < 3:
                self.session.fixation_cross.draw()

        elif self.phase == 7:  # iti
            self.session.fixation_cross.draw()

        super(LearningTrial, self).draw()


# class EndOfBlockTrial(MRITrial):
#
#     def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
#         super(EndOfBlockTrial, self).__init__(parameters=parameters,
#                                               phase_durations=phase_durations,
#                                               session=session,
#                                               screen=screen,
#                                               tracker=tracker)
#
#         self.ID = ID
#         self.parameters = parameters
#
#         # initialize times
#         self.t_time = self.jitter_time = self.stimulus_time = self.iti_time = None
#
#         # guestimate the amount of money to be earned
#         estimated_moneys = self.session.estimate_bonus()
#         end_str = '!' if estimated_moneys > 0 else ''
#
#         self.stim = [
#             visual.TextStim(screen, pos=(0, 4),
#                             units=self.session.config.get('text', 'units'),
#                             height=self.session.config.get('text', 'height'),
#                             text='End of block. So far, you earned %d points%s' % (self.session.total_points, end_str)),
#             visual.TextStim(screen, pos=(0, 0),
#                             units=self.session.config.get('text', 'units'),
#                             height=self.session.config.get('text', 'height'),
#                             text='Based on your performance so far, it looks like you will receive a bonus of approximately %.2f euro%s' % (estimated_moneys, end_str)),
#             visual.TextStim(screen, pos=(0, -4),
#                             units=self.session.config.get('text', 'units'),
#                             height=self.session.config.get('text', 'height'),
#                             text='You can take a short break now. Press <space bar> to continue.' %
#                                  self.session.total_points)
#         ]
#
#     def event(self):
#
#         for ev, time in event.getKeys(timeStamped=self.session.clock):
#             if len(ev) > 0:
#                 if ev in ['esc', 'escape']:
#                     self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
#                     self.stopped = True
#                     self.session.stopped = True
#                     print 'run canceled by user'
#
#                 # it handles both numeric and lettering modes
#                 elif ev == '+' or ev == 'equal':
#                     self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
#                     self.stopped = True
#                     print 'trial canceled by user'
#
#                 elif ev == self.session.mri_trigger_key:  # TR pulse
#                     self.events.append([99, time, self.session.clock.getTime() - self.start_time])
#                     # if self.phase == 0:
#                     #     self.phase_forward()
#
#                 elif ev in self.session.response_button_signs:
#                     self.events.append([ev, time, self.session.clock.getTime() - self.start_time, 'key_press'])
#
#                 elif ev in ['space']:
#                     self.events.append([ev, time, self.session.clock.getTime() - self.start_time, 'key_press'])
#                     if self.phase == 1:
#                         self.stopped = True
#
#             super(EndOfBlockTrial, self).key_event(ev)
#
#     def run(self):
#         """
#         Runs this trial
#         """
#
#         self.start_time = self.session.clock.getTime()
#         if self.tracker:
#             self.tracker.log('trial ' + str(self.ID) + ' started at ' + str(self.start_time) )
#             self.tracker.send_command('record_status_message "Trial ' + str(self.ID) + '"')
#         self.events.append('trial ' + str(self.ID) + ' started at ' + str(self.start_time))
#
#         while not self.stopped:
#             self.run_time = self.session.clock.getTime() - self.start_time
#
#             # waits for this phase to end (final pulse being collected)
#             if self.phase == 0:
#                 self.t_time = self.session.clock.getTime()
#                 if (self.t_time - self.start_time) > self.phase_durations[0]:
#                     self.phase_forward()
#
#             # events and draw, but only if we haven't stopped yet
#             if not self.stopped:
#                 self.event()
#                 self.draw()
#
#         self.stop()
#
#     def draw(self):
#
#         if self.phase == 1:   # waiting for scanner-time
#             for st in self.stim:
#                 st.draw()
#
#         super(EndOfBlockTrial, self).draw()
#
#
# class EndOfExperimentTrial(EndOfBlockTrial):
#     def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
#         super(EndOfExperimentTrial, self).__init__(ID=ID,
#                                                    parameters=parameters,
#                                                    phase_durations=phase_durations,
#                                                    session=session,
#                                                    screen=screen,
#                                                    tracker=tracker)
#
#         self.ID = ID
#         self.parameters = parameters
#
#         # initialize times
#         self.t_time = self.jitter_time = self.stimulus_time = self.iti_time = None
#
#         if self.session.practice:
#             txt = 'This is the end of the practice session\n\nPress <space bar> to continue to the real experiment'
#         else:
#             txt = 'This is the end of experiment\n\nPlease inform the experiment leader now'
#         self.stim = [
#             visual.TextStim(screen,
#                             pos=(0, 0),
#                             units=self.session.config.get('text', 'units'),
#                             height=self.session.config.get('text', 'height'),
#                             text=txt)]


class InstructionTrial(MRITrial):

    def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
        super(InstructionTrial, self).__init__(parameters=parameters,
                                               phase_durations=phase_durations,
                                               session=session,
                                               screen=screen,
                                               tracker=tracker)

        self.ID = ID
        self.parameters = parameters

        # initialize times
        self.t_time = self.stimulus_time = self.iti_time = None

#        self.instruction_text = visual.TextStim(screen, text='End of block. Waiting for operator...')

    def event(self):

        for ev, time in event.getKeys(timeStamped=self.session.clock):
            if len(ev) > 0:
                if ev in ['esc', 'escape']:
                    self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                    print 'run canceled by user'

                # it handles both numeric and lettering modes
                elif ev == '+' or ev == 'equal':
                    self.events.append([-99, time, self.session.clock.getTime() - self.start_time])
                    self.stopped = True
                    print 'trial canceled by user'

                elif ev == self.session.mri_trigger_key:  # TR pulse
                    self.events.append([99, time, self.session.clock.getTime() - self.start_time])
                    # if self.phase == 0:
                    #     self.phase_forward()

                elif ev in self.session.response_button_signs:
                    self.events.append([ev, time, self.session.clock.getTime() - self.start_time, 'key_press'])

                elif ev in ['space']:
                    self.events.append([ev, time, self.session.clock.getTime() - self.start_time, 'key_press'])
                    if self.phase == 1:
                        self.stopped = True

            super(InstructionTrial, self).key_event(ev)

    def run(self):
        """
        Runs this trial
        """

        self.start_time = self.session.clock.getTime()
        if self.tracker:
            self.tracker.log('trial ' + str(self.ID) + ' started at ' + str(self.start_time) )
            self.tracker.send_command('record_status_message "Trial ' + str(self.ID) + '"')
        self.events.append('trial ' + str(self.ID) + ' started at ' + str(self.start_time))

        while not self.stopped:
            self.run_time = self.session.clock.getTime() - self.start_time

            # waits for this phase to end (final pulse being collected)
            if self.phase == 0:
                self.t_time = self.session.clock.getTime()
                if (self.t_time - self.start_time) > self.phase_durations[0]:
                    self.phase_forward()

            # events and draw, but only if we haven't stopped yet
            if not self.stopped:
                self.event()
                self.draw()

        self.stop()

    def draw(self):

        for element in self.session.current_instruction_screen:
           element.draw()

        super(InstructionTrial, self).draw()


class PracticeTrial(LearningTrial):
    """ Practice trials do the same as normal trials, but annotate what happens in each different phase. """

    def __init__(self, ID, parameters, phase_durations, annotate=True, session=None, screen=None, tracker=None):
        super(PracticeTrial, self).__init__(ID, parameters, phase_durations, session=session, screen=screen, tracker=tracker)

        # prepare arrow
        arrow_left_vertices = [(0.2, 0.05),
                               (0.2, -0.05),
                               (0.0, -0.05),
                               (0, -0.1),
                               (-0.2, 0),
                               (0, 0.1),
                               (0, 0.05)]

        arrow_y_pos = self.session.config.get('text', 'feedback_y_pos')

        self.arrows = [visual.ShapeStim(win=screen,
                                        vertices=[(x*2+4, y*2+arrow_y_pos[0]) for x, y in arrow_left_vertices],
                                       fillColor='lightgray', size=1, lineColor='lightgray', units='deg'),

                       visual.ShapeStim(win=screen,
                                        vertices=[(x*2+4, y*2+arrow_y_pos[1]) for x, y in arrow_left_vertices],
                                        fillColor='lightgray', size=1, lineColor='lightgray', units='deg')]

        self.annotate = annotate
        if self.annotate:
            if parameters['cue'] == 'SPD':
                spdacc = 'fast'
            else:
                spdacc = 'accurate'
            self.forward_text = visual.TextStim(win=self.screen,
                                                text='Press <space bar> to continue\n(In the real task, this will go '
                                                     'automatically)',
                                                italic=True,
                                                pos=(0, -4), units='deg',
                                                wrapWidth=self.session.config.get('text', 'wrap_width'),
                                                alignHoriz='center'
                                                )
            self.annotations = {
                'phase_1': visual.TextStim(win=self.screen,
                                           text='This indicates the trial is about to start. Please keep your eyes '
                                                'fixed to this cross',
                                           pos=(0, 4), units='deg',
                                           wrapWidth=self.session.config.get('text', 'wrap_width')
                                           ),
                'phase_2': visual.TextStim(win=self.screen,
                                           text='This cue tells you to be fast or accurate. In the upcoming choice, '
                                                'you need to be %s!' % spdacc,
                                           pos=(0, 4), units='deg',
                                           wrapWidth=self.session.config.get('text', 'wrap_width')
                                           ),
                'phase_4': [
                    visual.TextStim(win=self.screen,
                                    text='These are your choice options',
                                    pos=(0, 4), units='deg'),
                    visual.TextStim(win=self.screen,
                                    text='Decide which you want to choose now!\nPress %s to choose left OR %s to '
                                         'choose right' %(self.session.response_button_signs[0],
                                                          self.session.response_button_signs[1]),
                                    pos=(0, -4), units='deg',
                                    wrapWidth=self.session.config.get('text', 'wrap_width')
                                    )],
                'phase_5': visual.TextStim(win=self.screen,
                                           text='Your choice is highlighted shortly',
                                           pos=(0, 4), units='deg'),
                'phase_6': [
                    self.arrows[0],
                    visual.TextStim(win=self.screen,
                                    text='This is the outcome of your choice',
                                    pos=(6, 0), units='deg',
                                    alignHoriz='left')],
                'phase_7': [
                    self.arrows[1],
                    visual.TextStim(win=self.screen,
                                    text='This is your reward. If you were in time, your reward is the '
                                         'same as the outcome. If you were too slow to decide, you get no '
                                         'points.',
                                    pos=(8, 0), units='deg', alignHoriz='left')],
                'phase_8': [
                    visual.TextStim(win=self.screen,
                                    text='This is what it looks like when you were too late. In these '
                                         'cases, you do not get points - even when the outcome was 100 '
                                         'points',
                                    pos=(8, 0), units='deg', alignHoriz='left',
                                    ),
                    visual.TextStim(win=self.screen,
                                    text='Press < B > if you want to restart this explanation.\n\nPress <space bar> '
                                         'to start practicing for real!',
                                    pos=(0, -4), units='deg', alignVert='top',
                                    wrapWidth=self.session.config.get('text', 'wrap_width')
                                    )]
            }

    def draw(self):

        if self.session.debug:
            self.update_debug_txt()
            self.session.debug_txt.draw()

        if self.phase == 0:   # waiting for scanner-time
            if self.parameters['block_trial_ID'] == 0:
                self.session.scanner_wait_screen.draw()
            else:
                self.session.fixation_cross.draw()

        elif self.phase == 1:  # Pre-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 2:  # Cue
            if self.current_cue is not None:
                self.current_cue.draw()

        elif self.phase == 3:  # Post-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 4:  # Stimulus
            for stim in self.stim:
                stim.draw()

            self.session.fixation_cross.draw()

        elif self.phase == 5:  # Selection
            for stim in self.stim:
                stim.draw()

            # self.stim.draw_selection_rect()
            self.session.fixation_cross.draw()

        elif self.phase in [6, 7, 8]:  # Feedback is now split in three phases, for annotations of different components
            for fb in self.current_feedback:
                fb.draw()

            if len(self.current_feedback) < 3:
                self.session.fixation_cross.draw()  # why does it draw this even with 'too slow' feedback?

        elif self.phase == 9:  # iti
            self.session.fixation_cross.draw()

        # Annotations
        if self.annotate:
            if 'phase_%d' %self.phase in self.annotations.keys():
                this_phase_annotations = self.annotations['phase_%d' % self.phase]
                if not isinstance(this_phase_annotations, list):
                    this_phase_annotations = [this_phase_annotations]
                for annotation in this_phase_annotations:
                    annotation.draw()

        if np.isinf(self.phase_durations[self.phase]) and self.phase not in [4, 8, 9]:
            self.forward_text.draw()

        super(PracticeTrial, self).draw()

    def event(self):

        for ev, time in event.getKeys(timeStamped=self.session.clock):
            if len(ev) > 0:
                if ev in ['minus', '-', 'b']:
                    self.events.append([-99, ev, time, self.session.clock.getTime() - self.start_time])
                    self.session.restart_block = True
                    self.stopped = True
                    print('block restarted by user')

                # Response (key press)
                elif ev in self.session.response_button_signs:
                    self.events.append([ev, time,  # absolute time since start of experiment
                                        self.session.clock.getTime() - self.start_time,  # time since start of trial
                                        self.stimulus_time - self.jitter_time_2,  # time since stimulus start
                                        'key_press'])

                    if self.phase == 4 or self.phase == 5:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time)

                            # Not in the MR scanner? End phase upon keypress
                            if self.session.tr == 0 and self.phase == 4:
                                self.phase_forward()

                elif ev in ['space'] and self.phase in [1, 2, 5, 6, 7, 8] and np.isinf(self.phase_durations[
                                                                                           self.phase]):
                    if self.phase == 7:
                        # adjust feedback
                        self.current_feedback[1] = self.session.feedback_earnings_objects[0]  # no points earned
                        self.current_feedback.append(self.session.feedback_timing_objects[0])  # too late

                    # if self.phase < 8:
                    self.phase_forward()
                    if self.phase == 2 and self.current_cue is None:
                        print('moving up!')
                        self.cue_time = self.session.clock.getTime()
                        self.phase_forward()  # move another phase forward
                    # else:
                    #     self.stopped = True

            super(PracticeTrial, self).event(ev, time)

    # def process_response(self, ev, time):
    #
    #     super(PracticeTrial, self).process_response(ev, time)
    #     if len(self.current_feedback)

    def run(self):
        """
        Runs this trial
        """

        self.start_time = self.session.clock.getTime()
        if self.tracker:
            self.tracker.log('trial ' + str(self.ID) + ' started at ' + str(self.start_time))
            self.tracker.send_command('record_status_message "Trial ' + str(self.ID) + '"')
        self.events.append('trial ' + str(self.ID) + ' started at ' + str(self.start_time))

        while not self.stopped:
            self.run_time = self.session.clock.getTime() - self.start_time

            # Waits for scanner pulse.
            if self.phase == 0:
                self.t_time = self.session.clock.getTime()
                if not isinstance(self.session, MRISession) or self.session.tr == 0:
                    self.phase_forward()

            # In phase 1, we show fix cross (jittered timing)
            if self.phase == 1:
                self.jitter_time_1 = self.session.clock.getTime()
                if (self.jitter_time_1 - self.t_time) > self.phase_durations[1]:
                    self.phase_forward()

            # In phase 2, we show the cue (fixed timing)
            if self.phase == 2:
                self.cue_time = self.session.clock.getTime()
                if (self.cue_time - self.jitter_time_1) > self.phase_durations[2]:
                    self.phase_forward()

            # In phase 3, we show the fixation cross again (jittered timing)
            if self.phase == 3:
                self.jitter_time_2 = self.session.clock.getTime()
                if (self.jitter_time_2 - self.cue_time) > self.phase_durations[3]:
                    self.phase_forward()

            # In phase 4, we show the stimulus
            if self.phase == 4:
                self.stimulus_time = self.session.clock.getTime()
                if (self.stimulus_time - self.jitter_time_2) > self.phase_durations[4]:
                    self.phase_forward()

            # In phase 5, we highlight the chosen option
            if self.phase == 5:
                self.selection_time = self.session.clock.getTime()
                if (self.selection_time - self.stimulus_time) > self.phase_durations[5]:
                    self.phase_forward()

            # In phase 6, we show feedback
            if self.phase == 6:
                self.feedback_time_1 = self.session.clock.getTime()
                if (self.feedback_time_1 - self.selection_time) > self.phase_durations[6]:
                    self.phase_forward()

            # In phase 7, we show feedback
            if self.phase == 7:
                self.feedback_time_2 = self.session.clock.getTime()
                if (self.feedback_time_2 - self.feedback_time_1) > self.phase_durations[7]:
                    self.phase_forward()

            # In phase 8, we show feedback
            if self.phase == 8:
                self.feedback_time_3 = self.session.clock.getTime()
                if (self.feedback_time_3 - self.feedback_time_2) > self.phase_durations[8]:
                    self.phase_forward()

            # ITI
            if self.phase == 9:
                self.iti_time = self.session.clock.getTime()
                if (self.iti_time - self.feedback_time_3) > self.phase_durations[9]:
                    self.phase_forward()
                    self.stopped = True

            # events and draw, but only if we haven't stopped yet
            if not self.stopped:
                self.event()
                self.draw()

                # # make screen shots
                # ss_fn = 'screenshots/trial_%d_phase_%d_set_%d.png' % (self.ID, self.phase, self.parameters[
                #     'stimulus_set'])
                # if not os.path.isfile(ss_fn):
                #     self.screen.getMovieFrame()  # Defaults to front buffer, I.e. what's on screen now.
                #     self.screen.saveMovieFrames(ss_fn)

        self.stop()
