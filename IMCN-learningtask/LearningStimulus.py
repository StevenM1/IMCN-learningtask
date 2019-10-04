from psychopy import visual
from copy import copy


class LearningStimulusSingle(object):
    """ Draws a single stimulus, instead of a set """

    def __init__(self, screen, stimulus, stimulus_type, x_pos, **kwargs):
        self.screen = screen
        self.x_pos = x_pos
        self.x_pos_current = x_pos
        self.selected = False

        if 'rect_line_width' in kwargs.keys():
            rect_line_width = kwargs['rect_line_width']
            kwargs.pop('rect_line_width')

        if stimulus_type == 'colors':
            kwargs.pop('text_height')
            self.stimulus = visual.Rect(self.screen, fillColor=stimulus, pos=(x_pos, 0), **kwargs)

        elif stimulus_type == 'agathodaimon':
            kws = copy(kwargs)
            print(kws)
            kws.pop('height')
            kws.pop('width')
            height = kws['text_height']
            kws.pop('text_height')
            self.stimulus = visual.TextStim(self.screen,
                                            text=stimulus,
                                            height=height,
                                            pos=(x_pos, 0),
                                            font='Agathodaimon',
                                            fontFiles=['./lib/AGATHODA.TTF'], **kws)

        height = kwargs['height']
        width = kwargs['width']
        kwargs.pop('height')
        kwargs.pop('width')
        kwargs.pop('text_height')

        self.selection_rect = visual.Rect(self.screen, width=width+.5, height=height+.5, fillColor=None,
                                         pos=(x_pos, 0), lineWidth=rect_line_width, **kwargs)

    def draw(self):

        self.stimulus.draw()

        if self.selected:
            self.selection_rect.draw()


class LearningStimulus(object):
    """ ToDo: document something about the chosen stimuli. Japanese signs, a la Michael Frank? Or colors?
    Agathodaimon alphabet seems to work
    """

    def __init__(self, screen, set, stimulus_type='colors', x_pos=(-1, 1), **kwargs):
        self.screen = screen
        self.x_pos = x_pos
        self.x_pos_current = x_pos
        self.selected = None

        if 'rect_line_width' in kwargs.keys():
            rect_line_width = kwargs['rect_line_width']
            kwargs.pop('rect_line_width')

        if stimulus_type == 'colors':
            kwargs.pop('text_height')
            self.stimuli = [
                visual.Rect(self.screen, fillColor=set[0], pos=(x_pos[0], 0), **kwargs),
                visual.Rect(self.screen, fillColor=set[1], pos=(x_pos[1], 0), **kwargs)
            ]
        elif stimulus_type == 'agathodaimon':
            kws = copy(kwargs)
            kws.pop('height')
            kws.pop('width')
            height = kws['text_height']
            kws.pop('text_height')
            self.stimuli = [
                visual.TextStim(self.screen, text=set[0], height=height, pos=(0, 0),  #pos=(x_pos[0], 0),
                                font='Agathodaimon',
                                fontFiles=['./lib/AGATHODA.TTF'], **kws),
                visual.TextStim(self.screen, text=set[1], height=height, pos=(0, 0),
                                font='Agathodaimon',
                                fontFiles=['./lib/AGATHODA.TTF'], **kws)
            ]

        height = kwargs['height']
        width = kwargs['width']
        kwargs.pop('height')
        kwargs.pop('width')
        kwargs.pop('text_height')

        self.selection_rect = [
            visual.Rect(self.screen, width=width+.5, height=height+.5, fillColor=None,
                        pos=(x_pos[0], 0), lineWidth=rect_line_width, **kwargs),
            visual.Rect(self.screen, width=width+.5, height=height+.5, fillColor=None,
                        pos=(x_pos[1], 0), lineWidth=rect_line_width, **kwargs)
        ]

    def reverse_order(self):
        """
        on half of all trials, we want to 'reverse' the presentation order of the stimuli to prevent any left/right
        biases
        """

        self.x_pos_corrent = self.x_pos[1], self.x_pos[0]

        for i in range(len(self.stimuli)):
            self.stimuli[i].pos = self.x_pos_current[i]
            # self.selection_rect[i].pos = self.x_pos_curent[i]

    def draw(self):
        for stim in self.stimuli:
            stim.draw()

    def draw_selection_rect(self):

        if self.selected is not None:
            self.selection_rect[self.selected].draw()


class FixationCross(object):
    """
    Fixation cross built according to recommendations in the following paper:

    Thaler, L., Schutz, A.C., Goodale, M.A., & Gegenfurtner, K.R. (2013). What is the best fixation target? The
    effect of target shape on stability of fixational eye movements. Vision research (2016), 76, 31-42.

    This small fixation cross combining a cross, bulls eye, and circle, apparently minimizes micro-saccadic movements
    during fixation.

    Parameters
    -----------
    win : psychopy.visual.Window instance
    outer_radius : float
        Radius of outer circle, in degrees of visual angle. Defaults to 0.15
    inner_radius : float
        Radius of inner circle (bulls eye), in degrees of visual angle. Defaults to 0.3
    bg : tuple
        RGB of background color. Defaults to (0.5, 0.5, 0.5) - gray screen.
    """

    def __init__(self, win, outer_radius=.3, inner_radius=.15, bg=(0.5, 0.5, 0.5)):

        self.fixation_circle = visual.Circle(win,
                                             radius=outer_radius,
                                             units='deg', fillColor='black', lineColor=bg)
        self.fixation_vertical_bar = visual.Rect(win, width=inner_radius, height=outer_radius * 2,
                                                 units='deg', fillColor=bg,
                                                 lineColor=bg)
        self.fixation_horizontal_bar = visual.Rect(win, width=outer_radius * 2, height=inner_radius,
                                                   units='deg', fillColor=bg,
                                                   lineColor=bg)
        self.fixation_bulls = visual.Circle(win, radius=inner_radius/2,
                                            units='deg', fillColor='black',
                                            lineColor='black')

    def draw(self):
        """
        Draws the fixation cross
        """

        self.fixation_circle.draw()
        self.fixation_vertical_bar.draw()
        self.fixation_horizontal_bar.draw()
        self.fixation_bulls.draw()
