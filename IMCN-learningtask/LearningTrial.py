from exptools.core.trial import MRITrial
from exptools.core.session import MRISession
from psychopy import event, visual
import numpy as np


class LearningTrial(MRITrial):

    def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
        super(LearningTrial, self).__init__(parameters=parameters,
                                            phase_durations=phase_durations,
                                            session=session,
                                            screen=screen,
                                            tracker=tracker)

        print(phase_durations)
        self.trs_recorded = 0
        self.ID = ID
        self.parameters = parameters

        # set-up response options / dict
        self.response_measured = False  # Has the pp responded yet?
        self.response = {}
        self.response['button'] = None
        self.response['rt'] = None
        self.response['side'] = None
        self.response['accuracy'] = None
        self.response['too_slow'] = None
        self.response['too_fast'] = None
        self.response['in_time'] = None

        # Select stimulus
        self.stim = self.session.stimuli[parameters['stimulus_set']]
        self.stim.selected = None  # make sure to reset

        # Select feedback
        self.points_earned = 0
        self.current_feedback = self.session.feedback_text_objects[0]  # default feedback (no response) is 'too slow'

        # Select cue
        if parameters['cue'] == 'SPD':
            self.current_cue = self.session.cues[0]
        elif parameters['cue'] == 'ACC':
            self.current_cue = self.session.cues[1]
        else:
            self.current_cue = None

        # initialize times
        self.t_time = self.jitter_time_1 = self.cue_time = self.jitter_time_2 = self.stimulus_time = \
            self.selection_time = self.feedback_time = self.iti_time = -1

    def event(self):

        for ev, time in event.getKeys(timeStamped=self.session.clock):
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

                    if self.phase == 4:
                        if not self.response_measured:
                            self.response_measured = True
                            self.process_response(ev, time)

                            # Not in the MR scanner? End phase upon keypress
                            if self.session.tr == 0:
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
        self.response['accuracy'] = ev == self.parameters['correct_response']
        self.response['too_slow'] = self.response['rt'] >= 2
        self.response['too_fast'] = self.response['rt'] <= .1
        self.response['in_time'] = 0.1 < self.response['rt'] < 2

        # to draw a rectangle
        self.stim.selected = self.response['side']

        # What is the outcome of the choice?
        p_choice = self.parameters['p_win'][self.response['side']]  # get probability of winning of current choice
        self.won = np.random.binomial(1, p_choice, 1)[0]  # did the participant win?

        # if the response was in time, the participant gains the reward
        if self.response['in_time']:
            self.points_earned = self.won

        # set-up feedback
        if self.response['too_slow'] and self.won:
            self.current_feedback = self.session.feedback_text_objects[1]
        elif self.response['too_slow'] and not self.won:
            self.current_feedback = self.session.feedback_text_objects[2]
        elif self.response['too_fast'] and self.won:
            self.current_feedback = self.session.feedback_text_objects[3]
        elif self.response['too_fast'] and not self.won:
            self.current_feedback = self.session.feedback_text_objects[4]
        elif self.won:  # won!
            self.current_feedback = self.session.feedback_text_objects[5]
        elif not self.won:  # Not won!
            self.current_feedback = self.session.feedback_text_objects[6]

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

        self.stop()

    def update_debug_txt(self):

        trial = self.ID
        phase = self.phase
        time = self.session.clock.getTime()
        trial_start_time = self.start_time
        t_start = time - trial_start_time
        stim = self.parameters['stimulus_set']
        ps = self.parameters['p_win']

        debug_str = 'Trial: %d\nPhase: %d\nTime since start: %.3fs\nTrial start time: %.3fs\nTime since trial start: %.3fs\nStim set: %s\nPs: [%.1f, %.1f]' %(
        trial, phase, time, trial_start_time, t_start, stim, ps[0], ps[1])
        resp_str = "\n".join(("{}: {}".format(*i) for i in self.response.items()))
        debug_str = debug_str + '\n' + resp_str
        self.session.debug_txt.text = debug_str

    def draw(self):

        if self.session.subject_initials == 'DEBUG':
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
            self.current_cue.draw()

        elif self.phase == 3:  # Post-cue fix cross
            self.session.fixation_cross.draw()

        elif self.phase == 4:  # Stimulus
            self.stim.draw()

        elif self.phase == 5:  # Selection
            self.stim.draw()
            self.stim.draw_selection_rect()

        elif self.phase == 6:  # feedback
            self.current_feedback.draw()

        elif self.phase == 7:  # iti
            self.session.fixation_cross.draw()

        super(LearningTrial, self).draw()


class EndOfBlockTrial(MRITrial):

    def __init__(self, ID, parameters, phase_durations, session=None, screen=None, tracker=None):
        super(EndOfBlockTrial, self).__init__(parameters=parameters,
                                              phase_durations=phase_durations,
                                              session=session,
                                              screen=screen,
                                              tracker=tracker)

        self.ID = ID
        self.parameters = parameters

        # initialize times
        self.t_time = self.jitter_time = self.stimulus_time = self.iti_time = None

        self.instruction_text = visual.TextStim(screen, text='End of block. Waiting for operator...')

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

            super(EndOfBlockTrial, self).key_event(ev)

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

        if self.phase == 1:   # waiting for scanner-time
            self.instruction_text.draw()

        super(EndOfBlockTrial, self).draw()
