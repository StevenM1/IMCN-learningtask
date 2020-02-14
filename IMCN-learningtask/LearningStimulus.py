from psychopy import visual
from copy import copy


class LearningStimulusSingle(object):
    """ Draws a single stimulus """

    def __init__(self, win, stimulus, stimulus_type, x_pos, **kwargs):
        self.win = win
        self.x_pos = x_pos
        self.x_pos_current = x_pos

        if stimulus_type == 'colors':
            _ = kwargs.pop('text_height')
            self.stimulus = visual.Rect(self.win, fillColor=stimulus, pos=(x_pos, 0), **kwargs)

        elif stimulus_type == 'symbol':
            kws = copy(kwargs)
            _ = kws.pop('height')
            _ = kws.pop('width')
            height = kws.pop('text_height')

            if stimulus in ['Z', 'z', 'q'] and kws['font'] == 'Agathodaimon':
                # These three symbols are too high/wide, decrease size
                height -= 1.3
            if kws['font'] == 'Glagolitsa':
                # This font is slightly "larger" than Agathodaimon, decrease size
                height *= 0.8

            self.stimulus = visual.TextStim(self.win,
                                            text=stimulus,
                                            height=height,
                                            alignHoriz='center',
                                            alignVert='center',
                                            pos=(x_pos, 0),
                                            **kws)

    def draw(self):

        self.stimulus.draw()


class SelectionRectangle(object):

    def __init__(self, win, x_pos, **kwargs):
        self.win = win
        rect_line_width = kwargs['rect_line_width']
        kwargs.pop('rect_line_width')
        height = kwargs['height']
        width = kwargs['width']
        kwargs.pop('height')
        kwargs.pop('width')
        kwargs.pop('text_height')

        self.selection_rect = visual.Rect(self.win,
                                          width=width+.5,
                                          height=height+.5, fillColor=None,
                                          pos=(x_pos, 0),
                                          lineWidth=rect_line_width,
                                          **kwargs)

    def draw(self):
        self.selection_rect.draw()


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


#
# class LearningStimulus(object):
#     """ ToDo: document something about the chosen stimuli. Japanese signs, a la Michael Frank? Or colors?
#     Agathodaimon alphabet seems to work
#     """
#
#     def __init__(self, win, set, stimulus_type='colors', x_pos=(-1, 1), **kwargs):
#         self.win = win
#         self.x_pos = x_pos
#         self.x_pos_current = x_pos
#         self.selected = None
#
#         if 'rect_line_width' in kwargs.keys():
#             rect_line_width = kwargs['rect_line_width']
#             kwargs.pop('rect_line_width')
#
#         if stimulus_type == 'colors':
#             kwargs.pop('text_height')
#             self.stimuli = [
#                 visual.Rect(self.win, fillColor=set[0], pos=(x_pos[0], 0), **kwargs),
#                 visual.Rect(self.win, fillColor=set[1], pos=(x_pos[1], 0), **kwargs)
#             ]
#         elif stimulus_type == 'agathodaimon':
#             kws = copy(kwargs)
#             kws.pop('height')
#             kws.pop('width')
#             height = kws['text_height']
#             kws.pop('text_height')
#             self.stimuli = [
#                 visual.TextStim(self.win, text=set[0], height=height, pos=(0, 0),  #pos=(x_pos[0], 0),
#                                 font='Agathodaimon',
#                                 fontFiles=['./lib/AGATHODA.TTF'], **kws),
#                 visual.TextStim(self.win, text=set[1], height=height, pos=(0, 0),
#                                 font='Agathodaimon',
#                                 fontFiles=['./lib/AGATHODA.TTF'], **kws)
#             ]
#
#         height = kwargs['height']
#         width = kwargs['width']
#         kwargs.pop('height')
#         kwargs.pop('width')
#         kwargs.pop('text_height')
#
#         self.selection_rect = [
#             visual.Rect(self.win, width=width+.5, height=height+.5, fillColor=None,
#                         pos=(x_pos[0], 0), lineWidth=rect_line_width, **kwargs),
#             visual.Rect(self.win, width=width+.5, height=height+.5, fillColor=None,
#                         pos=(x_pos[1], 0), lineWidth=rect_line_width, **kwargs)
#         ]
#
#     def reverse_order(self):
#         """
#         on half of all trials, we want to 'reverse' the presentation order of the stimuli to prevent any left/right
#         biases
#         """
#
#         self.x_pos_corrent = self.x_pos[1], self.x_pos[0]
#
#         for i in range(len(self.stimuli)):
#             self.stimuli[i].pos = self.x_pos_current[i]
#
#     def draw(self):
#         for stim in self.stimuli:
#             stim.draw()
#
#     def draw_selection_rect(self):
#
#         if self.selected is not None:
#             self.selection_rect[self.selected].draw()
