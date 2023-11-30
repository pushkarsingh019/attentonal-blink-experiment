#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Wed Nov 29 15:01:18 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'experiment'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/pushkarsingh/Documents/01 University/02 Experimental Psychology/02 project/02 experiment/experiment_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instructions" ---
    text_norm = visual.TextStim(win=win, name='text_norm',
        text="In this experiment, you will be shown a series of words and characters. At the end, you have to type the words in the series as best as you can remember.\n\n\n'press spacebar to continue'",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "practice_instructions" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text='This is the practice round.\n\nYou will be shown either characters like %%%%% or a word like "book" at a very fast speed. try to remember the word shown in the series.\n\nPress the spacebar to continue',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_2
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "fixation" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practice" ---
    distractor_1 = visual.TextStim(win=win, name='distractor_1',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor_2 = visual.TextStim(win=win, name='distractor_2',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_3 = visual.TextStim(win=win, name='distractor_3',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_4 = visual.TextStim(win=win, name='distractor_4',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    distractor_5 = visual.TextStim(win=win, name='distractor_5',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    target_1 = visual.TextStim(win=win, name='target_1',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_6 = visual.TextStim(win=win, name='distractor_6',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_7 = visual.TextStim(win=win, name='distractor_7',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    target_2 = visual.TextStim(win=win, name='target_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    distractor_8 = visual.TextStim(win=win, name='distractor_8',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    distractor_9 = visual.TextStim(win=win, name='distractor_9',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_10 = visual.TextStim(win=win, name='distractor_10',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "practice_response" ---
    textbox = visual.TextBox2(
         win, text=None, placeholder='1st word you remember', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.8, 0.2), borderWidth=1.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    practice_response_instructions = visual.TextStim(win=win, name='practice_response_instructions',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_2 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_2',
         depth=-3, autoLog=True,
    )
    button = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-4
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "instructions_testphase" ---
    instruction = visual.TextStim(win=win, name='instruction',
        text='Now we are moving on to test phase. \n\nIf you miss the word, try to make your best guess.\n\nPress the spacebar to continue',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_3
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "countdown" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fixation_2" ---
    cross_2 = visual.ShapeStim(
        win=win, name='cross_2', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "routine_1" ---
    distractor_75 = visual.TextStim(win=win, name='distractor_75',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor_76 = visual.TextStim(win=win, name='distractor_76',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_77 = visual.TextStim(win=win, name='distractor_77',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_78 = visual.TextStim(win=win, name='distractor_78',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    distractor_79 = visual.TextStim(win=win, name='distractor_79',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    target_14 = visual.TextStim(win=win, name='target_14',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_80 = visual.TextStim(win=win, name='distractor_80',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_81 = visual.TextStim(win=win, name='distractor_81',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    target_15 = visual.TextStim(win=win, name='target_15',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    distractor_82 = visual.TextStim(win=win, name='distractor_82',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    distractor_83 = visual.TextStim(win=win, name='distractor_83',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_84 = visual.TextStim(win=win, name='distractor_84',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "response" ---
    textbox_3 = visual.TextBox2(
         win, text=None, placeholder='first word you saw', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_3',
         depth=-1, autoLog=True,
    )
    practice_response_instructions_2 = visual.TextStim(win=win, name='practice_response_instructions_2',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_4 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_4',
         depth=-3, autoLog=True,
    )
    button_2 = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_2',
        depth=-4
    )
    button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "fixation_100ms" ---
    cross_3 = visual.ShapeStim(
        win=win, name='cross_3', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "routine_2" ---
    distractor_85 = visual.TextStim(win=win, name='distractor_85',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor_86 = visual.TextStim(win=win, name='distractor_86',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_87 = visual.TextStim(win=win, name='distractor_87',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_88 = visual.TextStim(win=win, name='distractor_88',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    distractor_89 = visual.TextStim(win=win, name='distractor_89',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    target_16 = visual.TextStim(win=win, name='target_16',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_90 = visual.TextStim(win=win, name='distractor_90',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_91 = visual.TextStim(win=win, name='distractor_91',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    target_17 = visual.TextStim(win=win, name='target_17',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    distractor_92 = visual.TextStim(win=win, name='distractor_92',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    distractor_93 = visual.TextStim(win=win, name='distractor_93',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_94 = visual.TextStim(win=win, name='distractor_94',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "response_100ms" ---
    textbox_5 = visual.TextBox2(
         win, text=None, placeholder='first word you saw', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_5',
         depth=-1, autoLog=True,
    )
    practice_response_instructions_3 = visual.TextStim(win=win, name='practice_response_instructions_3',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_6 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_6',
         depth=-3, autoLog=True,
    )
    button_3 = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_3',
        depth=-4
    )
    button_3.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "fixation" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "routine_34" ---
    distractor_95 = visual.TextStim(win=win, name='distractor_95',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor_96 = visual.TextStim(win=win, name='distractor_96',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_97 = visual.TextStim(win=win, name='distractor_97',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_98 = visual.TextStim(win=win, name='distractor_98',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    target_18 = visual.TextStim(win=win, name='target_18',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    distractor_99 = visual.TextStim(win=win, name='distractor_99',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_100 = visual.TextStim(win=win, name='distractor_100',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_101 = visual.TextStim(win=win, name='distractor_101',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    distractor_102 = visual.TextStim(win=win, name='distractor_102',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    target_19 = visual.TextStim(win=win, name='target_19',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    distractor_103 = visual.TextStim(win=win, name='distractor_103',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_104 = visual.TextStim(win=win, name='distractor_104',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "response_200ms" ---
    textbox_7 = visual.TextBox2(
         win, text=None, placeholder='first word you saw', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_7',
         depth=-1, autoLog=True,
    )
    practice_response_instructions_4 = visual.TextStim(win=win, name='practice_response_instructions_4',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_8 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_8',
         depth=-3, autoLog=True,
    )
    button_4 = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_4',
        depth=-4
    )
    button_4.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "fixation" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "routine_4" ---
    distractor_105 = visual.TextStim(win=win, name='distractor_105',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    target_20 = visual.TextStim(win=win, name='target_20',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_107 = visual.TextStim(win=win, name='distractor_107',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_108 = visual.TextStim(win=win, name='distractor_108',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    distractor_106 = visual.TextStim(win=win, name='distractor_106',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    distractor_109 = visual.TextStim(win=win, name='distractor_109',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_110 = visual.TextStim(win=win, name='distractor_110',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_111 = visual.TextStim(win=win, name='distractor_111',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    distractor_112 = visual.TextStim(win=win, name='distractor_112',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    distractor_113 = visual.TextStim(win=win, name='distractor_113',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    target_21 = visual.TextStim(win=win, name='target_21',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_114 = visual.TextStim(win=win, name='distractor_114',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    distractor_132 = visual.TextStim(win=win, name='distractor_132',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    distractor_133 = visual.TextStim(win=win, name='distractor_133',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    distractor_134 = visual.TextStim(win=win, name='distractor_134',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "response_400ms" ---
    textbox_9 = visual.TextBox2(
         win, text=None, placeholder='first word you saw', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_9',
         depth=-1, autoLog=True,
    )
    practice_response_instructions_5 = visual.TextStim(win=win, name='practice_response_instructions_5',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_10 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_10',
         depth=-3, autoLog=True,
    )
    button_5 = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_5',
        depth=-4
    )
    button_5.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "fixation" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "routine_5" ---
    distractor_115 = visual.TextStim(win=win, name='distractor_115',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    target_22 = visual.TextStim(win=win, name='target_22',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    distractor_116 = visual.TextStim(win=win, name='distractor_116',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    distractor_117 = visual.TextStim(win=win, name='distractor_117',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    distractor_118 = visual.TextStim(win=win, name='distractor_118',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    distractor_119 = visual.TextStim(win=win, name='distractor_119',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    distractor_120 = visual.TextStim(win=win, name='distractor_120',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    distractor_121 = visual.TextStim(win=win, name='distractor_121',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    distractor_122 = visual.TextStim(win=win, name='distractor_122',
        text='&&&&',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    distractor_123 = visual.TextStim(win=win, name='distractor_123',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    distractor_124 = visual.TextStim(win=win, name='distractor_124',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    distractor_125 = visual.TextStim(win=win, name='distractor_125',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    distractor_126 = visual.TextStim(win=win, name='distractor_126',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    distractor_127 = visual.TextStim(win=win, name='distractor_127',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    target_23 = visual.TextStim(win=win, name='target_23',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    distractor_128 = visual.TextStim(win=win, name='distractor_128',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-15.0);
    distractor_129 = visual.TextStim(win=win, name='distractor_129',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-16.0);
    distractor_130 = visual.TextStim(win=win, name='distractor_130',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-17.0);
    distractor_131 = visual.TextStim(win=win, name='distractor_131',
        text='%%%%',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-18.0);
    
    # --- Initialize components for Routine "response_600ms" ---
    textbox_11 = visual.TextBox2(
         win, text=None, placeholder='first word you saw', font='Arial',
         pos=(0, 0),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_11',
         depth=-1, autoLog=True,
    )
    practice_response_instructions_6 = visual.TextStim(win=win, name='practice_response_instructions_6',
        text='type the two words you saw as best as you can remember',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox_12 = visual.TextBox2(
         win, text=None, placeholder='second word', font='Arial',
         pos=(0, -0.1),     letterHeight=0.03,
         size=(0.5, 0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_12',
         depth=-3, autoLog=True,
    )
    button_6 = visual.ButtonStim(win, 
        text='submit', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_6',
        depth=-4
    )
    button_6.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "thank_you" ---
    text_norm_3 = visual.TextStim(win=win, name='text_norm_3',
        text='thank you for taking part :))\n\nPress the spacebar to exit',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_4 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_4
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime())
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    # keep track of which components have finished
    InstructionsComponents = [text_norm, key_instruct]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm* updates
        
        # if text_norm is starting this frame...
        if text_norm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm.frameNStart = frameN  # exact frame index
            text_norm.tStart = t  # local t and not account for scr refresh
            text_norm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm.status = STARTED
            text_norm.setAutoDraw(True)
        
        # if text_norm is active this frame...
        if text_norm.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct.started')
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime())
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_instructions.started', globalClock.getTime())
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # keep track of which components have finished
    practice_instructionsComponents = [text_norm_2, key_instruct_2]
    for thisComponent in practice_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_instructions" ---
    for thisComponent in practice_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_instructions.stopped', globalClock.getTime())
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "practice_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('practice.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime())
        # keep track of which components have finished
        fixationComponents = [cross]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross* updates
            
            # if cross is starting this frame...
            if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross.frameNStart = frameN  # exact frame index
                cross.tStart = t  # local t and not account for scr refresh
                cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.started')
                # update status
                cross.status = STARTED
                cross.setAutoDraw(True)
            
            # if cross is active this frame...
            if cross.status == STARTED:
                # update params
                pass
            
            # if cross is stopping this frame...
            if cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cross.tStop = t  # not accounting for scr refresh
                    cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.stopped')
                    # update status
                    cross.status = FINISHED
                    cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice.started', globalClock.getTime())
        target_1.setText(target1)
        target_2.setText(target2)
        # keep track of which components have finished
        practiceComponents = [distractor_1, distractor_2, distractor_3, distractor_4, distractor_5, target_1, distractor_6, distractor_7, target_2, distractor_8, distractor_9, distractor_10]
        for thisComponent in practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.636:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *distractor_1* updates
            
            # if distractor_1 is starting this frame...
            if distractor_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                distractor_1.frameNStart = frameN  # exact frame index
                distractor_1.tStart = t  # local t and not account for scr refresh
                distractor_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_1, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_1.status = STARTED
                distractor_1.setAutoDraw(True)
            
            # if distractor_1 is active this frame...
            if distractor_1.status == STARTED:
                # update params
                pass
            
            # if distractor_1 is stopping this frame...
            if distractor_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_1.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_1.tStop = t  # not accounting for scr refresh
                    distractor_1.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_1.status = FINISHED
                    distractor_1.setAutoDraw(False)
            
            # *distractor_2* updates
            
            # if distractor_2 is starting this frame...
            if distractor_2.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                # keep track of start time/frame for later
                distractor_2.frameNStart = frameN  # exact frame index
                distractor_2.tStart = t  # local t and not account for scr refresh
                distractor_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_2.status = STARTED
                distractor_2.setAutoDraw(True)
            
            # if distractor_2 is active this frame...
            if distractor_2.status == STARTED:
                # update params
                pass
            
            # if distractor_2 is stopping this frame...
            if distractor_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_2.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_2.tStop = t  # not accounting for scr refresh
                    distractor_2.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_2.status = FINISHED
                    distractor_2.setAutoDraw(False)
            
            # *distractor_3* updates
            
            # if distractor_3 is starting this frame...
            if distractor_3.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                # keep track of start time/frame for later
                distractor_3.frameNStart = frameN  # exact frame index
                distractor_3.tStart = t  # local t and not account for scr refresh
                distractor_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_3.status = STARTED
                distractor_3.setAutoDraw(True)
            
            # if distractor_3 is active this frame...
            if distractor_3.status == STARTED:
                # update params
                pass
            
            # if distractor_3 is stopping this frame...
            if distractor_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_3.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_3.tStop = t  # not accounting for scr refresh
                    distractor_3.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_3.status = FINISHED
                    distractor_3.setAutoDraw(False)
            
            # *distractor_4* updates
            
            # if distractor_4 is starting this frame...
            if distractor_4.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                # keep track of start time/frame for later
                distractor_4.frameNStart = frameN  # exact frame index
                distractor_4.tStart = t  # local t and not account for scr refresh
                distractor_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_4.status = STARTED
                distractor_4.setAutoDraw(True)
            
            # if distractor_4 is active this frame...
            if distractor_4.status == STARTED:
                # update params
                pass
            
            # if distractor_4 is stopping this frame...
            if distractor_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_4.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_4.tStop = t  # not accounting for scr refresh
                    distractor_4.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_4.status = FINISHED
                    distractor_4.setAutoDraw(False)
            
            # *distractor_5* updates
            
            # if distractor_5 is starting this frame...
            if distractor_5.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                # keep track of start time/frame for later
                distractor_5.frameNStart = frameN  # exact frame index
                distractor_5.tStart = t  # local t and not account for scr refresh
                distractor_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_5.status = STARTED
                distractor_5.setAutoDraw(True)
            
            # if distractor_5 is active this frame...
            if distractor_5.status == STARTED:
                # update params
                pass
            
            # if distractor_5 is stopping this frame...
            if distractor_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_5.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_5.tStop = t  # not accounting for scr refresh
                    distractor_5.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_5.status = FINISHED
                    distractor_5.setAutoDraw(False)
            
            # *target_1* updates
            
            # if target_1 is starting this frame...
            if target_1.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                # keep track of start time/frame for later
                target_1.frameNStart = frameN  # exact frame index
                target_1.tStart = t  # local t and not account for scr refresh
                target_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_1.started')
                # update status
                target_1.status = STARTED
                target_1.setAutoDraw(True)
            
            # if target_1 is active this frame...
            if target_1.status == STARTED:
                # update params
                pass
            
            # if target_1 is stopping this frame...
            if target_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_1.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    target_1.tStop = t  # not accounting for scr refresh
                    target_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_1.stopped')
                    # update status
                    target_1.status = FINISHED
                    target_1.setAutoDraw(False)
            
            # *distractor_6* updates
            
            # if distractor_6 is starting this frame...
            if distractor_6.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                # keep track of start time/frame for later
                distractor_6.frameNStart = frameN  # exact frame index
                distractor_6.tStart = t  # local t and not account for scr refresh
                distractor_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_6.status = STARTED
                distractor_6.setAutoDraw(True)
            
            # if distractor_6 is active this frame...
            if distractor_6.status == STARTED:
                # update params
                pass
            
            # if distractor_6 is stopping this frame...
            if distractor_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_6.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_6.tStop = t  # not accounting for scr refresh
                    distractor_6.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_6.status = FINISHED
                    distractor_6.setAutoDraw(False)
            
            # *distractor_7* updates
            
            # if distractor_7 is starting this frame...
            if distractor_7.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                # keep track of start time/frame for later
                distractor_7.frameNStart = frameN  # exact frame index
                distractor_7.tStart = t  # local t and not account for scr refresh
                distractor_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_7.status = STARTED
                distractor_7.setAutoDraw(True)
            
            # if distractor_7 is active this frame...
            if distractor_7.status == STARTED:
                # update params
                pass
            
            # if distractor_7 is stopping this frame...
            if distractor_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_7.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_7.tStop = t  # not accounting for scr refresh
                    distractor_7.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_7.status = FINISHED
                    distractor_7.setAutoDraw(False)
            
            # *target_2* updates
            
            # if target_2 is starting this frame...
            if target_2.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                # keep track of start time/frame for later
                target_2.frameNStart = frameN  # exact frame index
                target_2.tStart = t  # local t and not account for scr refresh
                target_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_2.started')
                # update status
                target_2.status = STARTED
                target_2.setAutoDraw(True)
            
            # if target_2 is active this frame...
            if target_2.status == STARTED:
                # update params
                pass
            
            # if target_2 is stopping this frame...
            if target_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_2.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    target_2.tStop = t  # not accounting for scr refresh
                    target_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_2.stopped')
                    # update status
                    target_2.status = FINISHED
                    target_2.setAutoDraw(False)
            
            # *distractor_8* updates
            
            # if distractor_8 is starting this frame...
            if distractor_8.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                # keep track of start time/frame for later
                distractor_8.frameNStart = frameN  # exact frame index
                distractor_8.tStart = t  # local t and not account for scr refresh
                distractor_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_8.status = STARTED
                distractor_8.setAutoDraw(True)
            
            # if distractor_8 is active this frame...
            if distractor_8.status == STARTED:
                # update params
                pass
            
            # if distractor_8 is stopping this frame...
            if distractor_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_8.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_8.tStop = t  # not accounting for scr refresh
                    distractor_8.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_8.status = FINISHED
                    distractor_8.setAutoDraw(False)
            
            # *distractor_9* updates
            
            # if distractor_9 is starting this frame...
            if distractor_9.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                # keep track of start time/frame for later
                distractor_9.frameNStart = frameN  # exact frame index
                distractor_9.tStart = t  # local t and not account for scr refresh
                distractor_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_9.status = STARTED
                distractor_9.setAutoDraw(True)
            
            # if distractor_9 is active this frame...
            if distractor_9.status == STARTED:
                # update params
                pass
            
            # if distractor_9 is stopping this frame...
            if distractor_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_9.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_9.tStop = t  # not accounting for scr refresh
                    distractor_9.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_9.status = FINISHED
                    distractor_9.setAutoDraw(False)
            
            # *distractor_10* updates
            
            # if distractor_10 is starting this frame...
            if distractor_10.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                # keep track of start time/frame for later
                distractor_10.frameNStart = frameN  # exact frame index
                distractor_10.tStart = t  # local t and not account for scr refresh
                distractor_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(distractor_10, 'tStartRefresh')  # time at next scr refresh
                # update status
                distractor_10.status = STARTED
                distractor_10.setAutoDraw(True)
            
            # if distractor_10 is active this frame...
            if distractor_10.status == STARTED:
                # update params
                pass
            
            # if distractor_10 is stopping this frame...
            if distractor_10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > distractor_10.tStartRefresh + 0.053-frameTolerance:
                    # keep track of stop time/frame for later
                    distractor_10.tStop = t  # not accounting for scr refresh
                    distractor_10.frameNStop = frameN  # exact frame index
                    # update status
                    distractor_10.status = FINISHED
                    distractor_10.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice" ---
        for thisComponent in practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.636000)
        
        # --- Prepare to start Routine "practice_response" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_response.started', globalClock.getTime())
        textbox.reset()
        textbox_2.reset()
        # reset button to account for continued clicks & clear times on/off
        button.reset()
        # keep track of which components have finished
        practice_responseComponents = [textbox, practice_response_instructions, textbox_2, button]
        for thisComponent in practice_responseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_response" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textbox* updates
            
            # if textbox is starting this frame...
            if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox.frameNStart = frameN  # exact frame index
                textbox.tStart = t  # local t and not account for scr refresh
                textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox.started')
                # update status
                textbox.status = STARTED
                textbox.setAutoDraw(True)
            
            # if textbox is active this frame...
            if textbox.status == STARTED:
                # update params
                pass
            
            # *practice_response_instructions* updates
            
            # if practice_response_instructions is starting this frame...
            if practice_response_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_response_instructions.frameNStart = frameN  # exact frame index
                practice_response_instructions.tStart = t  # local t and not account for scr refresh
                practice_response_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_response_instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_response_instructions.started')
                # update status
                practice_response_instructions.status = STARTED
                practice_response_instructions.setAutoDraw(True)
            
            # if practice_response_instructions is active this frame...
            if practice_response_instructions.status == STARTED:
                # update params
                pass
            
            # *textbox_2* updates
            
            # if textbox_2 is starting this frame...
            if textbox_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox_2.frameNStart = frameN  # exact frame index
                textbox_2.tStart = t  # local t and not account for scr refresh
                textbox_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox_2.started')
                # update status
                textbox_2.status = STARTED
                textbox_2.setAutoDraw(True)
            
            # if textbox_2 is active this frame...
            if textbox_2.status == STARTED:
                # update params
                pass
            # *button* updates
            
            # if button is starting this frame...
            if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button.frameNStart = frameN  # exact frame index
                button.tStart = t  # local t and not account for scr refresh
                button.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button.started')
                # update status
                button.status = STARTED
                button.setAutoDraw(True)
            
            # if button is active this frame...
            if button.status == STARTED:
                # update params
                pass
                # check whether button has been pressed
                if button.isClicked:
                    if not button.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button.timesOn.append(button.buttonClock.getTime())
                        button.timesOff.append(button.buttonClock.getTime())
                    elif len(button.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button.timesOff[-1] = button.buttonClock.getTime()
                    if not button.wasClicked:
                        # end routine when button is clicked
                        continueRoutine = False
                    if not button.wasClicked:
                        # run callback code when button is clicked
                        pass
            # take note of whether button was clicked, so that next frame we know if clicks are new
            button.wasClicked = button.isClicked and button.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_responseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_response" ---
        for thisComponent in practice_responseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_response.stopped', globalClock.getTime())
        # Run 'End Routine' code from accuracy_check
        if textbox.text == target1:
            thisExp.addData("f1_accuracy", 1)
        else:
            thisExp.addData("f1_accuracy", 0)
        
        if textbox_2.text == target2:
            thisExp.addData("f2_accuracy", 1)
        else:
            thisExp.addData("f2_accuracy", 0)
        
        
        trials.addData('textbox.text',textbox.text)
        trials.addData('textbox_2.text',textbox_2.text)
        trials.addData('button.numClicks', button.numClicks)
        if button.numClicks:
           trials.addData('button.timesOn', button.timesOn)
           trials.addData('button.timesOff', button.timesOff)
        else:
           trials.addData('button.timesOn', "")
           trials.addData('button.timesOff', "")
        # the Routine "practice_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "instructions_testphase" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_testphase.started', globalClock.getTime())
    key_instruct_3.keys = []
    key_instruct_3.rt = []
    _key_instruct_3_allKeys = []
    # keep track of which components have finished
    instructions_testphaseComponents = [instruction, key_instruct_3]
    for thisComponent in instructions_testphaseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_testphase" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction* updates
        
        # if instruction is starting this frame...
        if instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction.frameNStart = frameN  # exact frame index
            instruction.tStart = t  # local t and not account for scr refresh
            instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction.status = STARTED
            instruction.setAutoDraw(True)
        
        # if instruction is active this frame...
        if instruction.status == STARTED:
            # update params
            pass
        
        # *key_instruct_3* updates
        waitOnFlip = False
        
        # if key_instruct_3 is starting this frame...
        if key_instruct_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_3.frameNStart = frameN  # exact frame index
            key_instruct_3.tStart = t  # local t and not account for scr refresh
            key_instruct_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_3.started')
            # update status
            key_instruct_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_3_allKeys.extend(theseKeys)
            if len(_key_instruct_3_allKeys):
                key_instruct_3.keys = _key_instruct_3_allKeys[0].name  # just the first key pressed
                key_instruct_3.rt = _key_instruct_3_allKeys[0].rt
                key_instruct_3.duration = _key_instruct_3_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_testphaseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_testphase" ---
    for thisComponent in instructions_testphaseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_testphase.stopped', globalClock.getTime())
    # check responses
    if key_instruct_3.keys in ['', [], None]:  # No response was made
        key_instruct_3.keys = None
    thisExp.addData('key_instruct_3.keys',key_instruct_3.keys)
    if key_instruct_3.keys != None:  # we had a response
        thisExp.addData('key_instruct_3.rt', key_instruct_3.rt)
        thisExp.addData('key_instruct_3.duration', key_instruct_3.duration)
    thisExp.nextEntry()
    # the Routine "instructions_testphase" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "countdown" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('countdown.started', globalClock.getTime())
    # keep track of which components have finished
    countdownComponents = [text_countdown]
    for thisComponent in countdownComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "countdown" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_countdown* updates
        
        # if text_countdown is starting this frame...
        if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_countdown.frameNStart = frameN  # exact frame index
            text_countdown.tStart = t  # local t and not account for scr refresh
            text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_countdown.started')
            # update status
            text_countdown.status = STARTED
            text_countdown.setAutoDraw(True)
        
        # if text_countdown is active this frame...
        if text_countdown.status == STARTED:
            # update params
            text_countdown.setText(str(10-int(t)), log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in countdownComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "countdown" ---
    for thisComponent in countdownComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('countdown.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # set up handler to look after randomisation of conditions etc
    counter_balance = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='counter_balance')
    thisExp.addLoop(counter_balance)  # add the loop to the experiment
    thisCounter_balance = counter_balance.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCounter_balance.rgb)
    if thisCounter_balance != None:
        for paramName in thisCounter_balance:
            globals()[paramName] = thisCounter_balance[paramName]
    
    for thisCounter_balance in counter_balance:
        currentLoop = counter_balance
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisCounter_balance.rgb)
        if thisCounter_balance != None:
            for paramName in thisCounter_balance:
                globals()[paramName] = thisCounter_balance[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_50ms = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('50ms.xlsx'),
            seed=None, name='trials_50ms')
        thisExp.addLoop(trials_50ms)  # add the loop to the experiment
        thisTrials_50m = trials_50ms.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_50m.rgb)
        if thisTrials_50m != None:
            for paramName in thisTrials_50m:
                globals()[paramName] = thisTrials_50m[paramName]
        
        for thisTrials_50m in trials_50ms:
            currentLoop = trials_50ms
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_50m.rgb)
            if thisTrials_50m != None:
                for paramName in thisTrials_50m:
                    globals()[paramName] = thisTrials_50m[paramName]
            
            # --- Prepare to start Routine "fixation_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_2.started', globalClock.getTime())
            # keep track of which components have finished
            fixation_2Components = [cross_2]
            for thisComponent in fixation_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_2* updates
                
                # if cross_2 is starting this frame...
                if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_2.frameNStart = frameN  # exact frame index
                    cross_2.tStart = t  # local t and not account for scr refresh
                    cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_2.started')
                    # update status
                    cross_2.status = STARTED
                    cross_2.setAutoDraw(True)
                
                # if cross_2 is active this frame...
                if cross_2.status == STARTED:
                    # update params
                    pass
                
                # if cross_2 is stopping this frame...
                if cross_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross_2.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cross_2.tStop = t  # not accounting for scr refresh
                        cross_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_2.stopped')
                        # update status
                        cross_2.status = FINISHED
                        cross_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_2" ---
            for thisComponent in fixation_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_2.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "routine_1" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('routine_1.started', globalClock.getTime())
            target_14.setText(target1)
            target_15.setText(target2)
            # keep track of which components have finished
            routine_1Components = [distractor_75, distractor_76, distractor_77, distractor_78, distractor_79, target_14, distractor_80, distractor_81, target_15, distractor_82, distractor_83, distractor_84]
            for thisComponent in routine_1Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "routine_1" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.636:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *distractor_75* updates
                
                # if distractor_75 is starting this frame...
                if distractor_75.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_75.frameNStart = frameN  # exact frame index
                    distractor_75.tStart = t  # local t and not account for scr refresh
                    distractor_75.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_75, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_75.status = STARTED
                    distractor_75.setAutoDraw(True)
                
                # if distractor_75 is active this frame...
                if distractor_75.status == STARTED:
                    # update params
                    pass
                
                # if distractor_75 is stopping this frame...
                if distractor_75.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_75.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_75.tStop = t  # not accounting for scr refresh
                        distractor_75.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_75.status = FINISHED
                        distractor_75.setAutoDraw(False)
                
                # *distractor_76* updates
                
                # if distractor_76 is starting this frame...
                if distractor_76.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_76.frameNStart = frameN  # exact frame index
                    distractor_76.tStart = t  # local t and not account for scr refresh
                    distractor_76.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_76, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_76.status = STARTED
                    distractor_76.setAutoDraw(True)
                
                # if distractor_76 is active this frame...
                if distractor_76.status == STARTED:
                    # update params
                    pass
                
                # if distractor_76 is stopping this frame...
                if distractor_76.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_76.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_76.tStop = t  # not accounting for scr refresh
                        distractor_76.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_76.status = FINISHED
                        distractor_76.setAutoDraw(False)
                
                # *distractor_77* updates
                
                # if distractor_77 is starting this frame...
                if distractor_77.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_77.frameNStart = frameN  # exact frame index
                    distractor_77.tStart = t  # local t and not account for scr refresh
                    distractor_77.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_77, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_77.status = STARTED
                    distractor_77.setAutoDraw(True)
                
                # if distractor_77 is active this frame...
                if distractor_77.status == STARTED:
                    # update params
                    pass
                
                # if distractor_77 is stopping this frame...
                if distractor_77.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_77.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_77.tStop = t  # not accounting for scr refresh
                        distractor_77.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_77.status = FINISHED
                        distractor_77.setAutoDraw(False)
                
                # *distractor_78* updates
                
                # if distractor_78 is starting this frame...
                if distractor_78.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_78.frameNStart = frameN  # exact frame index
                    distractor_78.tStart = t  # local t and not account for scr refresh
                    distractor_78.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_78, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_78.status = STARTED
                    distractor_78.setAutoDraw(True)
                
                # if distractor_78 is active this frame...
                if distractor_78.status == STARTED:
                    # update params
                    pass
                
                # if distractor_78 is stopping this frame...
                if distractor_78.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_78.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_78.tStop = t  # not accounting for scr refresh
                        distractor_78.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_78.status = FINISHED
                        distractor_78.setAutoDraw(False)
                
                # *distractor_79* updates
                
                # if distractor_79 is starting this frame...
                if distractor_79.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_79.frameNStart = frameN  # exact frame index
                    distractor_79.tStart = t  # local t and not account for scr refresh
                    distractor_79.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_79, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_79.status = STARTED
                    distractor_79.setAutoDraw(True)
                
                # if distractor_79 is active this frame...
                if distractor_79.status == STARTED:
                    # update params
                    pass
                
                # if distractor_79 is stopping this frame...
                if distractor_79.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_79.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_79.tStop = t  # not accounting for scr refresh
                        distractor_79.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_79.status = FINISHED
                        distractor_79.setAutoDraw(False)
                
                # *target_14* updates
                
                # if target_14 is starting this frame...
                if target_14.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                    # keep track of start time/frame for later
                    target_14.frameNStart = frameN  # exact frame index
                    target_14.tStart = t  # local t and not account for scr refresh
                    target_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_14, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_14.started')
                    # update status
                    target_14.status = STARTED
                    target_14.setAutoDraw(True)
                
                # if target_14 is active this frame...
                if target_14.status == STARTED:
                    # update params
                    pass
                
                # if target_14 is stopping this frame...
                if target_14.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_14.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_14.tStop = t  # not accounting for scr refresh
                        target_14.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_14.stopped')
                        # update status
                        target_14.status = FINISHED
                        target_14.setAutoDraw(False)
                
                # *distractor_80* updates
                
                # if distractor_80 is starting this frame...
                if distractor_80.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_80.frameNStart = frameN  # exact frame index
                    distractor_80.tStart = t  # local t and not account for scr refresh
                    distractor_80.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_80, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_80.status = STARTED
                    distractor_80.setAutoDraw(True)
                
                # if distractor_80 is active this frame...
                if distractor_80.status == STARTED:
                    # update params
                    pass
                
                # if distractor_80 is stopping this frame...
                if distractor_80.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_80.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_80.tStop = t  # not accounting for scr refresh
                        distractor_80.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_80.status = FINISHED
                        distractor_80.setAutoDraw(False)
                
                # *distractor_81* updates
                
                # if distractor_81 is starting this frame...
                if distractor_81.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_81.frameNStart = frameN  # exact frame index
                    distractor_81.tStart = t  # local t and not account for scr refresh
                    distractor_81.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_81, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_81.status = STARTED
                    distractor_81.setAutoDraw(True)
                
                # if distractor_81 is active this frame...
                if distractor_81.status == STARTED:
                    # update params
                    pass
                
                # if distractor_81 is stopping this frame...
                if distractor_81.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_81.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_81.tStop = t  # not accounting for scr refresh
                        distractor_81.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_81.status = FINISHED
                        distractor_81.setAutoDraw(False)
                
                # *target_15* updates
                
                # if target_15 is starting this frame...
                if target_15.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                    # keep track of start time/frame for later
                    target_15.frameNStart = frameN  # exact frame index
                    target_15.tStart = t  # local t and not account for scr refresh
                    target_15.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_15, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_15.started')
                    # update status
                    target_15.status = STARTED
                    target_15.setAutoDraw(True)
                
                # if target_15 is active this frame...
                if target_15.status == STARTED:
                    # update params
                    pass
                
                # if target_15 is stopping this frame...
                if target_15.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_15.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_15.tStop = t  # not accounting for scr refresh
                        target_15.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_15.stopped')
                        # update status
                        target_15.status = FINISHED
                        target_15.setAutoDraw(False)
                
                # *distractor_82* updates
                
                # if distractor_82 is starting this frame...
                if distractor_82.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_82.frameNStart = frameN  # exact frame index
                    distractor_82.tStart = t  # local t and not account for scr refresh
                    distractor_82.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_82, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_82.status = STARTED
                    distractor_82.setAutoDraw(True)
                
                # if distractor_82 is active this frame...
                if distractor_82.status == STARTED:
                    # update params
                    pass
                
                # if distractor_82 is stopping this frame...
                if distractor_82.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_82.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_82.tStop = t  # not accounting for scr refresh
                        distractor_82.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_82.status = FINISHED
                        distractor_82.setAutoDraw(False)
                
                # *distractor_83* updates
                
                # if distractor_83 is starting this frame...
                if distractor_83.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_83.frameNStart = frameN  # exact frame index
                    distractor_83.tStart = t  # local t and not account for scr refresh
                    distractor_83.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_83, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_83.status = STARTED
                    distractor_83.setAutoDraw(True)
                
                # if distractor_83 is active this frame...
                if distractor_83.status == STARTED:
                    # update params
                    pass
                
                # if distractor_83 is stopping this frame...
                if distractor_83.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_83.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_83.tStop = t  # not accounting for scr refresh
                        distractor_83.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_83.status = FINISHED
                        distractor_83.setAutoDraw(False)
                
                # *distractor_84* updates
                
                # if distractor_84 is starting this frame...
                if distractor_84.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_84.frameNStart = frameN  # exact frame index
                    distractor_84.tStart = t  # local t and not account for scr refresh
                    distractor_84.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_84, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_84.status = STARTED
                    distractor_84.setAutoDraw(True)
                
                # if distractor_84 is active this frame...
                if distractor_84.status == STARTED:
                    # update params
                    pass
                
                # if distractor_84 is stopping this frame...
                if distractor_84.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_84.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_84.tStop = t  # not accounting for scr refresh
                        distractor_84.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_84.status = FINISHED
                        distractor_84.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_1" ---
            for thisComponent in routine_1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('routine_1.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.636000)
            
            # --- Prepare to start Routine "response" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('response.started', globalClock.getTime())
            textbox_3.reset()
            textbox_4.reset()
            # reset button_2 to account for continued clicks & clear times on/off
            button_2.reset()
            # keep track of which components have finished
            responseComponents = [textbox_3, practice_response_instructions_2, textbox_4, button_2]
            for thisComponent in responseComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textbox_3* updates
                
                # if textbox_3 is starting this frame...
                if textbox_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_3.frameNStart = frameN  # exact frame index
                    textbox_3.tStart = t  # local t and not account for scr refresh
                    textbox_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_3.started')
                    # update status
                    textbox_3.status = STARTED
                    textbox_3.setAutoDraw(True)
                
                # if textbox_3 is active this frame...
                if textbox_3.status == STARTED:
                    # update params
                    pass
                
                # *practice_response_instructions_2* updates
                
                # if practice_response_instructions_2 is starting this frame...
                if practice_response_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    practice_response_instructions_2.frameNStart = frameN  # exact frame index
                    practice_response_instructions_2.tStart = t  # local t and not account for scr refresh
                    practice_response_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(practice_response_instructions_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_response_instructions_2.started')
                    # update status
                    practice_response_instructions_2.status = STARTED
                    practice_response_instructions_2.setAutoDraw(True)
                
                # if practice_response_instructions_2 is active this frame...
                if practice_response_instructions_2.status == STARTED:
                    # update params
                    pass
                
                # *textbox_4* updates
                
                # if textbox_4 is starting this frame...
                if textbox_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_4.frameNStart = frameN  # exact frame index
                    textbox_4.tStart = t  # local t and not account for scr refresh
                    textbox_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_4.started')
                    # update status
                    textbox_4.status = STARTED
                    textbox_4.setAutoDraw(True)
                
                # if textbox_4 is active this frame...
                if textbox_4.status == STARTED:
                    # update params
                    pass
                # *button_2* updates
                
                # if button_2 is starting this frame...
                if button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_2.frameNStart = frameN  # exact frame index
                    button_2.tStart = t  # local t and not account for scr refresh
                    button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_2.started')
                    # update status
                    button_2.status = STARTED
                    button_2.setAutoDraw(True)
                
                # if button_2 is active this frame...
                if button_2.status == STARTED:
                    # update params
                    pass
                    # check whether button_2 has been pressed
                    if button_2.isClicked:
                        if not button_2.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_2.timesOn.append(button_2.buttonClock.getTime())
                            button_2.timesOff.append(button_2.buttonClock.getTime())
                        elif len(button_2.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_2.timesOff[-1] = button_2.buttonClock.getTime()
                        if not button_2.wasClicked:
                            # end routine when button_2 is clicked
                            continueRoutine = False
                        if not button_2.wasClicked:
                            # run callback code when button_2 is clicked
                            pass
                # take note of whether button_2 was clicked, so that next frame we know if clicks are new
                button_2.wasClicked = button_2.isClicked and button_2.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in responseComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response" ---
            for thisComponent in responseComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('response.stopped', globalClock.getTime())
            # Run 'End Routine' code from accuracy_check_2
            if textbox_3.text == target1:
                thisExp.addData("f1_accuracy", 1)
            else:
                thisExp.addData("f1_accuracy", 0)
            
            if textbox_4.text == target2:
                thisExp.addData("f2_accuracy", 1)
            else:
                thisExp.addData("f2_accuracy", 0)
            
            
            trials_50ms.addData('textbox_3.text',textbox_3.text)
            trials_50ms.addData('textbox_4.text',textbox_4.text)
            trials_50ms.addData('button_2.numClicks', button_2.numClicks)
            if button_2.numClicks:
               trials_50ms.addData('button_2.timesOn', button_2.timesOn)
               trials_50ms.addData('button_2.timesOff', button_2.timesOff)
            else:
               trials_50ms.addData('button_2.timesOn', "")
               trials_50ms.addData('button_2.timesOff', "")
            # the Routine "response" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_50ms'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_100ms = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('100ms.xlsx'),
            seed=None, name='trials_100ms')
        thisExp.addLoop(trials_100ms)  # add the loop to the experiment
        thisTrials_100m = trials_100ms.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_100m.rgb)
        if thisTrials_100m != None:
            for paramName in thisTrials_100m:
                globals()[paramName] = thisTrials_100m[paramName]
        
        for thisTrials_100m in trials_100ms:
            currentLoop = trials_100ms
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_100m.rgb)
            if thisTrials_100m != None:
                for paramName in thisTrials_100m:
                    globals()[paramName] = thisTrials_100m[paramName]
            
            # --- Prepare to start Routine "fixation_100ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_100ms.started', globalClock.getTime())
            # keep track of which components have finished
            fixation_100msComponents = [cross_3]
            for thisComponent in fixation_100msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_100ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_3* updates
                
                # if cross_3 is starting this frame...
                if cross_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_3.frameNStart = frameN  # exact frame index
                    cross_3.tStart = t  # local t and not account for scr refresh
                    cross_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_3.started')
                    # update status
                    cross_3.status = STARTED
                    cross_3.setAutoDraw(True)
                
                # if cross_3 is active this frame...
                if cross_3.status == STARTED:
                    # update params
                    pass
                
                # if cross_3 is stopping this frame...
                if cross_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross_3.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cross_3.tStop = t  # not accounting for scr refresh
                        cross_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_3.stopped')
                        # update status
                        cross_3.status = FINISHED
                        cross_3.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_100msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_100ms" ---
            for thisComponent in fixation_100msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_100ms.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "routine_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('routine_2.started', globalClock.getTime())
            target_16.setText(target1)
            target_17.setText(target2)
            # keep track of which components have finished
            routine_2Components = [distractor_85, distractor_86, distractor_87, distractor_88, distractor_89, target_16, distractor_90, distractor_91, target_17, distractor_92, distractor_93, distractor_94]
            for thisComponent in routine_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "routine_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.636:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *distractor_85* updates
                
                # if distractor_85 is starting this frame...
                if distractor_85.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_85.frameNStart = frameN  # exact frame index
                    distractor_85.tStart = t  # local t and not account for scr refresh
                    distractor_85.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_85, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_85.status = STARTED
                    distractor_85.setAutoDraw(True)
                
                # if distractor_85 is active this frame...
                if distractor_85.status == STARTED:
                    # update params
                    pass
                
                # if distractor_85 is stopping this frame...
                if distractor_85.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_85.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_85.tStop = t  # not accounting for scr refresh
                        distractor_85.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_85.status = FINISHED
                        distractor_85.setAutoDraw(False)
                
                # *distractor_86* updates
                
                # if distractor_86 is starting this frame...
                if distractor_86.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_86.frameNStart = frameN  # exact frame index
                    distractor_86.tStart = t  # local t and not account for scr refresh
                    distractor_86.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_86, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_86.status = STARTED
                    distractor_86.setAutoDraw(True)
                
                # if distractor_86 is active this frame...
                if distractor_86.status == STARTED:
                    # update params
                    pass
                
                # if distractor_86 is stopping this frame...
                if distractor_86.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_86.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_86.tStop = t  # not accounting for scr refresh
                        distractor_86.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_86.status = FINISHED
                        distractor_86.setAutoDraw(False)
                
                # *distractor_87* updates
                
                # if distractor_87 is starting this frame...
                if distractor_87.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_87.frameNStart = frameN  # exact frame index
                    distractor_87.tStart = t  # local t and not account for scr refresh
                    distractor_87.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_87, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_87.status = STARTED
                    distractor_87.setAutoDraw(True)
                
                # if distractor_87 is active this frame...
                if distractor_87.status == STARTED:
                    # update params
                    pass
                
                # if distractor_87 is stopping this frame...
                if distractor_87.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_87.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_87.tStop = t  # not accounting for scr refresh
                        distractor_87.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_87.status = FINISHED
                        distractor_87.setAutoDraw(False)
                
                # *distractor_88* updates
                
                # if distractor_88 is starting this frame...
                if distractor_88.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_88.frameNStart = frameN  # exact frame index
                    distractor_88.tStart = t  # local t and not account for scr refresh
                    distractor_88.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_88, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_88.status = STARTED
                    distractor_88.setAutoDraw(True)
                
                # if distractor_88 is active this frame...
                if distractor_88.status == STARTED:
                    # update params
                    pass
                
                # if distractor_88 is stopping this frame...
                if distractor_88.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_88.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_88.tStop = t  # not accounting for scr refresh
                        distractor_88.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_88.status = FINISHED
                        distractor_88.setAutoDraw(False)
                
                # *distractor_89* updates
                
                # if distractor_89 is starting this frame...
                if distractor_89.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_89.frameNStart = frameN  # exact frame index
                    distractor_89.tStart = t  # local t and not account for scr refresh
                    distractor_89.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_89, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_89.status = STARTED
                    distractor_89.setAutoDraw(True)
                
                # if distractor_89 is active this frame...
                if distractor_89.status == STARTED:
                    # update params
                    pass
                
                # if distractor_89 is stopping this frame...
                if distractor_89.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_89.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_89.tStop = t  # not accounting for scr refresh
                        distractor_89.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_89.status = FINISHED
                        distractor_89.setAutoDraw(False)
                
                # *target_16* updates
                
                # if target_16 is starting this frame...
                if target_16.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                    # keep track of start time/frame for later
                    target_16.frameNStart = frameN  # exact frame index
                    target_16.tStart = t  # local t and not account for scr refresh
                    target_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_16, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_16.started')
                    # update status
                    target_16.status = STARTED
                    target_16.setAutoDraw(True)
                
                # if target_16 is active this frame...
                if target_16.status == STARTED:
                    # update params
                    pass
                
                # if target_16 is stopping this frame...
                if target_16.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_16.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_16.tStop = t  # not accounting for scr refresh
                        target_16.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_16.stopped')
                        # update status
                        target_16.status = FINISHED
                        target_16.setAutoDraw(False)
                
                # *distractor_90* updates
                
                # if distractor_90 is starting this frame...
                if distractor_90.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_90.frameNStart = frameN  # exact frame index
                    distractor_90.tStart = t  # local t and not account for scr refresh
                    distractor_90.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_90, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_90.status = STARTED
                    distractor_90.setAutoDraw(True)
                
                # if distractor_90 is active this frame...
                if distractor_90.status == STARTED:
                    # update params
                    pass
                
                # if distractor_90 is stopping this frame...
                if distractor_90.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_90.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_90.tStop = t  # not accounting for scr refresh
                        distractor_90.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_90.status = FINISHED
                        distractor_90.setAutoDraw(False)
                
                # *distractor_91* updates
                
                # if distractor_91 is starting this frame...
                if distractor_91.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_91.frameNStart = frameN  # exact frame index
                    distractor_91.tStart = t  # local t and not account for scr refresh
                    distractor_91.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_91, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_91.status = STARTED
                    distractor_91.setAutoDraw(True)
                
                # if distractor_91 is active this frame...
                if distractor_91.status == STARTED:
                    # update params
                    pass
                
                # if distractor_91 is stopping this frame...
                if distractor_91.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_91.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_91.tStop = t  # not accounting for scr refresh
                        distractor_91.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_91.status = FINISHED
                        distractor_91.setAutoDraw(False)
                
                # *target_17* updates
                
                # if target_17 is starting this frame...
                if target_17.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                    # keep track of start time/frame for later
                    target_17.frameNStart = frameN  # exact frame index
                    target_17.tStart = t  # local t and not account for scr refresh
                    target_17.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_17, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_17.started')
                    # update status
                    target_17.status = STARTED
                    target_17.setAutoDraw(True)
                
                # if target_17 is active this frame...
                if target_17.status == STARTED:
                    # update params
                    pass
                
                # if target_17 is stopping this frame...
                if target_17.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_17.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_17.tStop = t  # not accounting for scr refresh
                        target_17.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_17.stopped')
                        # update status
                        target_17.status = FINISHED
                        target_17.setAutoDraw(False)
                
                # *distractor_92* updates
                
                # if distractor_92 is starting this frame...
                if distractor_92.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_92.frameNStart = frameN  # exact frame index
                    distractor_92.tStart = t  # local t and not account for scr refresh
                    distractor_92.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_92, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_92.status = STARTED
                    distractor_92.setAutoDraw(True)
                
                # if distractor_92 is active this frame...
                if distractor_92.status == STARTED:
                    # update params
                    pass
                
                # if distractor_92 is stopping this frame...
                if distractor_92.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_92.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_92.tStop = t  # not accounting for scr refresh
                        distractor_92.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_92.status = FINISHED
                        distractor_92.setAutoDraw(False)
                
                # *distractor_93* updates
                
                # if distractor_93 is starting this frame...
                if distractor_93.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_93.frameNStart = frameN  # exact frame index
                    distractor_93.tStart = t  # local t and not account for scr refresh
                    distractor_93.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_93, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_93.status = STARTED
                    distractor_93.setAutoDraw(True)
                
                # if distractor_93 is active this frame...
                if distractor_93.status == STARTED:
                    # update params
                    pass
                
                # if distractor_93 is stopping this frame...
                if distractor_93.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_93.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_93.tStop = t  # not accounting for scr refresh
                        distractor_93.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_93.status = FINISHED
                        distractor_93.setAutoDraw(False)
                
                # *distractor_94* updates
                
                # if distractor_94 is starting this frame...
                if distractor_94.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_94.frameNStart = frameN  # exact frame index
                    distractor_94.tStart = t  # local t and not account for scr refresh
                    distractor_94.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_94, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_94.status = STARTED
                    distractor_94.setAutoDraw(True)
                
                # if distractor_94 is active this frame...
                if distractor_94.status == STARTED:
                    # update params
                    pass
                
                # if distractor_94 is stopping this frame...
                if distractor_94.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_94.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_94.tStop = t  # not accounting for scr refresh
                        distractor_94.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_94.status = FINISHED
                        distractor_94.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_2" ---
            for thisComponent in routine_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('routine_2.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.636000)
            
            # --- Prepare to start Routine "response_100ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('response_100ms.started', globalClock.getTime())
            textbox_5.reset()
            textbox_6.reset()
            # reset button_3 to account for continued clicks & clear times on/off
            button_3.reset()
            # keep track of which components have finished
            response_100msComponents = [textbox_5, practice_response_instructions_3, textbox_6, button_3]
            for thisComponent in response_100msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response_100ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textbox_5* updates
                
                # if textbox_5 is starting this frame...
                if textbox_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_5.frameNStart = frameN  # exact frame index
                    textbox_5.tStart = t  # local t and not account for scr refresh
                    textbox_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_5.started')
                    # update status
                    textbox_5.status = STARTED
                    textbox_5.setAutoDraw(True)
                
                # if textbox_5 is active this frame...
                if textbox_5.status == STARTED:
                    # update params
                    pass
                
                # *practice_response_instructions_3* updates
                
                # if practice_response_instructions_3 is starting this frame...
                if practice_response_instructions_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    practice_response_instructions_3.frameNStart = frameN  # exact frame index
                    practice_response_instructions_3.tStart = t  # local t and not account for scr refresh
                    practice_response_instructions_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(practice_response_instructions_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_response_instructions_3.started')
                    # update status
                    practice_response_instructions_3.status = STARTED
                    practice_response_instructions_3.setAutoDraw(True)
                
                # if practice_response_instructions_3 is active this frame...
                if practice_response_instructions_3.status == STARTED:
                    # update params
                    pass
                
                # *textbox_6* updates
                
                # if textbox_6 is starting this frame...
                if textbox_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_6.frameNStart = frameN  # exact frame index
                    textbox_6.tStart = t  # local t and not account for scr refresh
                    textbox_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_6.started')
                    # update status
                    textbox_6.status = STARTED
                    textbox_6.setAutoDraw(True)
                
                # if textbox_6 is active this frame...
                if textbox_6.status == STARTED:
                    # update params
                    pass
                # *button_3* updates
                
                # if button_3 is starting this frame...
                if button_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_3.frameNStart = frameN  # exact frame index
                    button_3.tStart = t  # local t and not account for scr refresh
                    button_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_3.started')
                    # update status
                    button_3.status = STARTED
                    button_3.setAutoDraw(True)
                
                # if button_3 is active this frame...
                if button_3.status == STARTED:
                    # update params
                    pass
                    # check whether button_3 has been pressed
                    if button_3.isClicked:
                        if not button_3.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_3.timesOn.append(button_3.buttonClock.getTime())
                            button_3.timesOff.append(button_3.buttonClock.getTime())
                        elif len(button_3.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_3.timesOff[-1] = button_3.buttonClock.getTime()
                        if not button_3.wasClicked:
                            # end routine when button_3 is clicked
                            continueRoutine = False
                        if not button_3.wasClicked:
                            # run callback code when button_3 is clicked
                            pass
                # take note of whether button_3 was clicked, so that next frame we know if clicks are new
                button_3.wasClicked = button_3.isClicked and button_3.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in response_100msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response_100ms" ---
            for thisComponent in response_100msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('response_100ms.stopped', globalClock.getTime())
            # Run 'End Routine' code from accuracy_check_3
            if textbox_5.text == target1:
                thisExp.addData("f1_accuracy", 1)
            else:
                thisExp.addData("f1_accuracy", 0)
            
            if textbox_6.text == target2:
                thisExp.addData("f2_accuracy", 1)
            else:
                thisExp.addData("f2_accuracy", 0)
            
            
            trials_100ms.addData('textbox_5.text',textbox_5.text)
            trials_100ms.addData('textbox_6.text',textbox_6.text)
            trials_100ms.addData('button_3.numClicks', button_3.numClicks)
            if button_3.numClicks:
               trials_100ms.addData('button_3.timesOn', button_3.timesOn)
               trials_100ms.addData('button_3.timesOff', button_3.timesOff)
            else:
               trials_100ms.addData('button_3.timesOn', "")
               trials_100ms.addData('button_3.timesOff', "")
            # the Routine "response_100ms" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_100ms'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_200ms = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('200ms.xlsx'),
            seed=None, name='trials_200ms')
        thisExp.addLoop(trials_200ms)  # add the loop to the experiment
        thisTrials_200m = trials_200ms.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_200m.rgb)
        if thisTrials_200m != None:
            for paramName in thisTrials_200m:
                globals()[paramName] = thisTrials_200m[paramName]
        
        for thisTrials_200m in trials_200ms:
            currentLoop = trials_200ms
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_200m.rgb)
            if thisTrials_200m != None:
                for paramName in thisTrials_200m:
                    globals()[paramName] = thisTrials_200m[paramName]
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # keep track of which components have finished
            fixationComponents = [cross]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross* updates
                
                # if cross is starting this frame...
                if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross.frameNStart = frameN  # exact frame index
                    cross.tStart = t  # local t and not account for scr refresh
                    cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.started')
                    # update status
                    cross.status = STARTED
                    cross.setAutoDraw(True)
                
                # if cross is active this frame...
                if cross.status == STARTED:
                    # update params
                    pass
                
                # if cross is stopping this frame...
                if cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        # update status
                        cross.status = FINISHED
                        cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "routine_34" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('routine_34.started', globalClock.getTime())
            target_18.setText(target1)
            target_19.setText(target2)
            # keep track of which components have finished
            routine_34Components = [distractor_95, distractor_96, distractor_97, distractor_98, target_18, distractor_99, distractor_100, distractor_101, distractor_102, target_19, distractor_103, distractor_104]
            for thisComponent in routine_34Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "routine_34" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.636:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *distractor_95* updates
                
                # if distractor_95 is starting this frame...
                if distractor_95.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_95.frameNStart = frameN  # exact frame index
                    distractor_95.tStart = t  # local t and not account for scr refresh
                    distractor_95.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_95, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_95.status = STARTED
                    distractor_95.setAutoDraw(True)
                
                # if distractor_95 is active this frame...
                if distractor_95.status == STARTED:
                    # update params
                    pass
                
                # if distractor_95 is stopping this frame...
                if distractor_95.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_95.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_95.tStop = t  # not accounting for scr refresh
                        distractor_95.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_95.status = FINISHED
                        distractor_95.setAutoDraw(False)
                
                # *distractor_96* updates
                
                # if distractor_96 is starting this frame...
                if distractor_96.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_96.frameNStart = frameN  # exact frame index
                    distractor_96.tStart = t  # local t and not account for scr refresh
                    distractor_96.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_96, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_96.status = STARTED
                    distractor_96.setAutoDraw(True)
                
                # if distractor_96 is active this frame...
                if distractor_96.status == STARTED:
                    # update params
                    pass
                
                # if distractor_96 is stopping this frame...
                if distractor_96.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_96.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_96.tStop = t  # not accounting for scr refresh
                        distractor_96.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_96.status = FINISHED
                        distractor_96.setAutoDraw(False)
                
                # *distractor_97* updates
                
                # if distractor_97 is starting this frame...
                if distractor_97.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_97.frameNStart = frameN  # exact frame index
                    distractor_97.tStart = t  # local t and not account for scr refresh
                    distractor_97.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_97, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_97.status = STARTED
                    distractor_97.setAutoDraw(True)
                
                # if distractor_97 is active this frame...
                if distractor_97.status == STARTED:
                    # update params
                    pass
                
                # if distractor_97 is stopping this frame...
                if distractor_97.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_97.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_97.tStop = t  # not accounting for scr refresh
                        distractor_97.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_97.status = FINISHED
                        distractor_97.setAutoDraw(False)
                
                # *distractor_98* updates
                
                # if distractor_98 is starting this frame...
                if distractor_98.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_98.frameNStart = frameN  # exact frame index
                    distractor_98.tStart = t  # local t and not account for scr refresh
                    distractor_98.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_98, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_98.status = STARTED
                    distractor_98.setAutoDraw(True)
                
                # if distractor_98 is active this frame...
                if distractor_98.status == STARTED:
                    # update params
                    pass
                
                # if distractor_98 is stopping this frame...
                if distractor_98.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_98.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_98.tStop = t  # not accounting for scr refresh
                        distractor_98.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_98.status = FINISHED
                        distractor_98.setAutoDraw(False)
                
                # *target_18* updates
                
                # if target_18 is starting this frame...
                if target_18.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                    # keep track of start time/frame for later
                    target_18.frameNStart = frameN  # exact frame index
                    target_18.tStart = t  # local t and not account for scr refresh
                    target_18.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_18, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_18.started')
                    # update status
                    target_18.status = STARTED
                    target_18.setAutoDraw(True)
                
                # if target_18 is active this frame...
                if target_18.status == STARTED:
                    # update params
                    pass
                
                # if target_18 is stopping this frame...
                if target_18.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_18.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_18.tStop = t  # not accounting for scr refresh
                        target_18.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_18.stopped')
                        # update status
                        target_18.status = FINISHED
                        target_18.setAutoDraw(False)
                
                # *distractor_99* updates
                
                # if distractor_99 is starting this frame...
                if distractor_99.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_99.frameNStart = frameN  # exact frame index
                    distractor_99.tStart = t  # local t and not account for scr refresh
                    distractor_99.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_99, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_99.status = STARTED
                    distractor_99.setAutoDraw(True)
                
                # if distractor_99 is active this frame...
                if distractor_99.status == STARTED:
                    # update params
                    pass
                
                # if distractor_99 is stopping this frame...
                if distractor_99.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_99.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_99.tStop = t  # not accounting for scr refresh
                        distractor_99.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_99.status = FINISHED
                        distractor_99.setAutoDraw(False)
                
                # *distractor_100* updates
                
                # if distractor_100 is starting this frame...
                if distractor_100.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_100.frameNStart = frameN  # exact frame index
                    distractor_100.tStart = t  # local t and not account for scr refresh
                    distractor_100.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_100, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_100.status = STARTED
                    distractor_100.setAutoDraw(True)
                
                # if distractor_100 is active this frame...
                if distractor_100.status == STARTED:
                    # update params
                    pass
                
                # if distractor_100 is stopping this frame...
                if distractor_100.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_100.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_100.tStop = t  # not accounting for scr refresh
                        distractor_100.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_100.status = FINISHED
                        distractor_100.setAutoDraw(False)
                
                # *distractor_101* updates
                
                # if distractor_101 is starting this frame...
                if distractor_101.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_101.frameNStart = frameN  # exact frame index
                    distractor_101.tStart = t  # local t and not account for scr refresh
                    distractor_101.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_101, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_101.status = STARTED
                    distractor_101.setAutoDraw(True)
                
                # if distractor_101 is active this frame...
                if distractor_101.status == STARTED:
                    # update params
                    pass
                
                # if distractor_101 is stopping this frame...
                if distractor_101.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_101.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_101.tStop = t  # not accounting for scr refresh
                        distractor_101.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_101.status = FINISHED
                        distractor_101.setAutoDraw(False)
                
                # *distractor_102* updates
                
                # if distractor_102 is starting this frame...
                if distractor_102.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_102.frameNStart = frameN  # exact frame index
                    distractor_102.tStart = t  # local t and not account for scr refresh
                    distractor_102.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_102, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_102.status = STARTED
                    distractor_102.setAutoDraw(True)
                
                # if distractor_102 is active this frame...
                if distractor_102.status == STARTED:
                    # update params
                    pass
                
                # if distractor_102 is stopping this frame...
                if distractor_102.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_102.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_102.tStop = t  # not accounting for scr refresh
                        distractor_102.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_102.status = FINISHED
                        distractor_102.setAutoDraw(False)
                
                # *target_19* updates
                
                # if target_19 is starting this frame...
                if target_19.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                    # keep track of start time/frame for later
                    target_19.frameNStart = frameN  # exact frame index
                    target_19.tStart = t  # local t and not account for scr refresh
                    target_19.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_19, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_19.started')
                    # update status
                    target_19.status = STARTED
                    target_19.setAutoDraw(True)
                
                # if target_19 is active this frame...
                if target_19.status == STARTED:
                    # update params
                    pass
                
                # if target_19 is stopping this frame...
                if target_19.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_19.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_19.tStop = t  # not accounting for scr refresh
                        target_19.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_19.stopped')
                        # update status
                        target_19.status = FINISHED
                        target_19.setAutoDraw(False)
                
                # *distractor_103* updates
                
                # if distractor_103 is starting this frame...
                if distractor_103.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_103.frameNStart = frameN  # exact frame index
                    distractor_103.tStart = t  # local t and not account for scr refresh
                    distractor_103.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_103, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_103.status = STARTED
                    distractor_103.setAutoDraw(True)
                
                # if distractor_103 is active this frame...
                if distractor_103.status == STARTED:
                    # update params
                    pass
                
                # if distractor_103 is stopping this frame...
                if distractor_103.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_103.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_103.tStop = t  # not accounting for scr refresh
                        distractor_103.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_103.status = FINISHED
                        distractor_103.setAutoDraw(False)
                
                # *distractor_104* updates
                
                # if distractor_104 is starting this frame...
                if distractor_104.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_104.frameNStart = frameN  # exact frame index
                    distractor_104.tStart = t  # local t and not account for scr refresh
                    distractor_104.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_104, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_104.status = STARTED
                    distractor_104.setAutoDraw(True)
                
                # if distractor_104 is active this frame...
                if distractor_104.status == STARTED:
                    # update params
                    pass
                
                # if distractor_104 is stopping this frame...
                if distractor_104.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_104.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_104.tStop = t  # not accounting for scr refresh
                        distractor_104.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_104.status = FINISHED
                        distractor_104.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_34Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_34" ---
            for thisComponent in routine_34Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('routine_34.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.636000)
            
            # --- Prepare to start Routine "response_200ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('response_200ms.started', globalClock.getTime())
            textbox_7.reset()
            textbox_8.reset()
            # reset button_4 to account for continued clicks & clear times on/off
            button_4.reset()
            # keep track of which components have finished
            response_200msComponents = [textbox_7, practice_response_instructions_4, textbox_8, button_4]
            for thisComponent in response_200msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response_200ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textbox_7* updates
                
                # if textbox_7 is starting this frame...
                if textbox_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_7.frameNStart = frameN  # exact frame index
                    textbox_7.tStart = t  # local t and not account for scr refresh
                    textbox_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_7.started')
                    # update status
                    textbox_7.status = STARTED
                    textbox_7.setAutoDraw(True)
                
                # if textbox_7 is active this frame...
                if textbox_7.status == STARTED:
                    # update params
                    pass
                
                # *practice_response_instructions_4* updates
                
                # if practice_response_instructions_4 is starting this frame...
                if practice_response_instructions_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    practice_response_instructions_4.frameNStart = frameN  # exact frame index
                    practice_response_instructions_4.tStart = t  # local t and not account for scr refresh
                    practice_response_instructions_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(practice_response_instructions_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_response_instructions_4.started')
                    # update status
                    practice_response_instructions_4.status = STARTED
                    practice_response_instructions_4.setAutoDraw(True)
                
                # if practice_response_instructions_4 is active this frame...
                if practice_response_instructions_4.status == STARTED:
                    # update params
                    pass
                
                # *textbox_8* updates
                
                # if textbox_8 is starting this frame...
                if textbox_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_8.frameNStart = frameN  # exact frame index
                    textbox_8.tStart = t  # local t and not account for scr refresh
                    textbox_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_8.started')
                    # update status
                    textbox_8.status = STARTED
                    textbox_8.setAutoDraw(True)
                
                # if textbox_8 is active this frame...
                if textbox_8.status == STARTED:
                    # update params
                    pass
                # *button_4* updates
                
                # if button_4 is starting this frame...
                if button_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_4.frameNStart = frameN  # exact frame index
                    button_4.tStart = t  # local t and not account for scr refresh
                    button_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_4.started')
                    # update status
                    button_4.status = STARTED
                    button_4.setAutoDraw(True)
                
                # if button_4 is active this frame...
                if button_4.status == STARTED:
                    # update params
                    pass
                    # check whether button_4 has been pressed
                    if button_4.isClicked:
                        if not button_4.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_4.timesOn.append(button_4.buttonClock.getTime())
                            button_4.timesOff.append(button_4.buttonClock.getTime())
                        elif len(button_4.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_4.timesOff[-1] = button_4.buttonClock.getTime()
                        if not button_4.wasClicked:
                            # end routine when button_4 is clicked
                            continueRoutine = False
                        if not button_4.wasClicked:
                            # run callback code when button_4 is clicked
                            pass
                # take note of whether button_4 was clicked, so that next frame we know if clicks are new
                button_4.wasClicked = button_4.isClicked and button_4.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in response_200msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response_200ms" ---
            for thisComponent in response_200msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('response_200ms.stopped', globalClock.getTime())
            # Run 'End Routine' code from accuracy_check_4
            if textbox_7.text == target1:
                thisExp.addData("f1_accuracy", 1)
            else:
                thisExp.addData("f1_accuracy", 0)
            
            if textbox_8.text == target2:
                thisExp.addData("f2_accuracy", 1)
            else:
                thisExp.addData("f2_accuracy", 0)
            
            
            trials_200ms.addData('textbox_7.text',textbox_7.text)
            trials_200ms.addData('textbox_8.text',textbox_8.text)
            trials_200ms.addData('button_4.numClicks', button_4.numClicks)
            if button_4.numClicks:
               trials_200ms.addData('button_4.timesOn', button_4.timesOn)
               trials_200ms.addData('button_4.timesOff', button_4.timesOff)
            else:
               trials_200ms.addData('button_4.timesOn', "")
               trials_200ms.addData('button_4.timesOff', "")
            # the Routine "response_200ms" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_200ms'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_400ms = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('400ms.xlsx'),
            seed=None, name='trials_400ms')
        thisExp.addLoop(trials_400ms)  # add the loop to the experiment
        thisTrials_400m = trials_400ms.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_400m.rgb)
        if thisTrials_400m != None:
            for paramName in thisTrials_400m:
                globals()[paramName] = thisTrials_400m[paramName]
        
        for thisTrials_400m in trials_400ms:
            currentLoop = trials_400ms
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_400m.rgb)
            if thisTrials_400m != None:
                for paramName in thisTrials_400m:
                    globals()[paramName] = thisTrials_400m[paramName]
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # keep track of which components have finished
            fixationComponents = [cross]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross* updates
                
                # if cross is starting this frame...
                if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross.frameNStart = frameN  # exact frame index
                    cross.tStart = t  # local t and not account for scr refresh
                    cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.started')
                    # update status
                    cross.status = STARTED
                    cross.setAutoDraw(True)
                
                # if cross is active this frame...
                if cross.status == STARTED:
                    # update params
                    pass
                
                # if cross is stopping this frame...
                if cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        # update status
                        cross.status = FINISHED
                        cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "routine_4" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('routine_4.started', globalClock.getTime())
            target_20.setText(target1)
            target_21.setText(target2)
            # keep track of which components have finished
            routine_4Components = [distractor_105, target_20, distractor_107, distractor_108, distractor_106, distractor_109, distractor_110, distractor_111, distractor_112, distractor_113, target_21, distractor_114, distractor_132, distractor_133, distractor_134]
            for thisComponent in routine_4Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "routine_4" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.795:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *distractor_105* updates
                
                # if distractor_105 is starting this frame...
                if distractor_105.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_105.frameNStart = frameN  # exact frame index
                    distractor_105.tStart = t  # local t and not account for scr refresh
                    distractor_105.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_105, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_105.status = STARTED
                    distractor_105.setAutoDraw(True)
                
                # if distractor_105 is active this frame...
                if distractor_105.status == STARTED:
                    # update params
                    pass
                
                # if distractor_105 is stopping this frame...
                if distractor_105.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_105.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_105.tStop = t  # not accounting for scr refresh
                        distractor_105.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_105.status = FINISHED
                        distractor_105.setAutoDraw(False)
                
                # *target_20* updates
                
                # if target_20 is starting this frame...
                if target_20.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                    # keep track of start time/frame for later
                    target_20.frameNStart = frameN  # exact frame index
                    target_20.tStart = t  # local t and not account for scr refresh
                    target_20.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_20, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_20.started')
                    # update status
                    target_20.status = STARTED
                    target_20.setAutoDraw(True)
                
                # if target_20 is active this frame...
                if target_20.status == STARTED:
                    # update params
                    pass
                
                # if target_20 is stopping this frame...
                if target_20.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_20.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_20.tStop = t  # not accounting for scr refresh
                        target_20.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_20.stopped')
                        # update status
                        target_20.status = FINISHED
                        target_20.setAutoDraw(False)
                
                # *distractor_107* updates
                
                # if distractor_107 is starting this frame...
                if distractor_107.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_107.frameNStart = frameN  # exact frame index
                    distractor_107.tStart = t  # local t and not account for scr refresh
                    distractor_107.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_107, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_107.status = STARTED
                    distractor_107.setAutoDraw(True)
                
                # if distractor_107 is active this frame...
                if distractor_107.status == STARTED:
                    # update params
                    pass
                
                # if distractor_107 is stopping this frame...
                if distractor_107.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_107.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_107.tStop = t  # not accounting for scr refresh
                        distractor_107.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_107.status = FINISHED
                        distractor_107.setAutoDraw(False)
                
                # *distractor_108* updates
                
                # if distractor_108 is starting this frame...
                if distractor_108.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_108.frameNStart = frameN  # exact frame index
                    distractor_108.tStart = t  # local t and not account for scr refresh
                    distractor_108.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_108, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_108.status = STARTED
                    distractor_108.setAutoDraw(True)
                
                # if distractor_108 is active this frame...
                if distractor_108.status == STARTED:
                    # update params
                    pass
                
                # if distractor_108 is stopping this frame...
                if distractor_108.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_108.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_108.tStop = t  # not accounting for scr refresh
                        distractor_108.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_108.status = FINISHED
                        distractor_108.setAutoDraw(False)
                
                # *distractor_106* updates
                
                # if distractor_106 is starting this frame...
                if distractor_106.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_106.frameNStart = frameN  # exact frame index
                    distractor_106.tStart = t  # local t and not account for scr refresh
                    distractor_106.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_106, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_106.status = STARTED
                    distractor_106.setAutoDraw(True)
                
                # if distractor_106 is active this frame...
                if distractor_106.status == STARTED:
                    # update params
                    pass
                
                # if distractor_106 is stopping this frame...
                if distractor_106.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_106.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_106.tStop = t  # not accounting for scr refresh
                        distractor_106.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_106.status = FINISHED
                        distractor_106.setAutoDraw(False)
                
                # *distractor_109* updates
                
                # if distractor_109 is starting this frame...
                if distractor_109.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_109.frameNStart = frameN  # exact frame index
                    distractor_109.tStart = t  # local t and not account for scr refresh
                    distractor_109.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_109, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_109.status = STARTED
                    distractor_109.setAutoDraw(True)
                
                # if distractor_109 is active this frame...
                if distractor_109.status == STARTED:
                    # update params
                    pass
                
                # if distractor_109 is stopping this frame...
                if distractor_109.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_109.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_109.tStop = t  # not accounting for scr refresh
                        distractor_109.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_109.status = FINISHED
                        distractor_109.setAutoDraw(False)
                
                # *distractor_110* updates
                
                # if distractor_110 is starting this frame...
                if distractor_110.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_110.frameNStart = frameN  # exact frame index
                    distractor_110.tStart = t  # local t and not account for scr refresh
                    distractor_110.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_110, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_110.status = STARTED
                    distractor_110.setAutoDraw(True)
                
                # if distractor_110 is active this frame...
                if distractor_110.status == STARTED:
                    # update params
                    pass
                
                # if distractor_110 is stopping this frame...
                if distractor_110.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_110.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_110.tStop = t  # not accounting for scr refresh
                        distractor_110.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_110.status = FINISHED
                        distractor_110.setAutoDraw(False)
                
                # *distractor_111* updates
                
                # if distractor_111 is starting this frame...
                if distractor_111.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_111.frameNStart = frameN  # exact frame index
                    distractor_111.tStart = t  # local t and not account for scr refresh
                    distractor_111.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_111, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_111.status = STARTED
                    distractor_111.setAutoDraw(True)
                
                # if distractor_111 is active this frame...
                if distractor_111.status == STARTED:
                    # update params
                    pass
                
                # if distractor_111 is stopping this frame...
                if distractor_111.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_111.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_111.tStop = t  # not accounting for scr refresh
                        distractor_111.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_111.status = FINISHED
                        distractor_111.setAutoDraw(False)
                
                # *distractor_112* updates
                
                # if distractor_112 is starting this frame...
                if distractor_112.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_112.frameNStart = frameN  # exact frame index
                    distractor_112.tStart = t  # local t and not account for scr refresh
                    distractor_112.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_112, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_112.status = STARTED
                    distractor_112.setAutoDraw(True)
                
                # if distractor_112 is active this frame...
                if distractor_112.status == STARTED:
                    # update params
                    pass
                
                # if distractor_112 is stopping this frame...
                if distractor_112.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_112.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_112.tStop = t  # not accounting for scr refresh
                        distractor_112.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_112.status = FINISHED
                        distractor_112.setAutoDraw(False)
                
                # *distractor_113* updates
                
                # if distractor_113 is starting this frame...
                if distractor_113.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_113.frameNStart = frameN  # exact frame index
                    distractor_113.tStart = t  # local t and not account for scr refresh
                    distractor_113.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_113, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_113.status = STARTED
                    distractor_113.setAutoDraw(True)
                
                # if distractor_113 is active this frame...
                if distractor_113.status == STARTED:
                    # update params
                    pass
                
                # if distractor_113 is stopping this frame...
                if distractor_113.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_113.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_113.tStop = t  # not accounting for scr refresh
                        distractor_113.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_113.status = FINISHED
                        distractor_113.setAutoDraw(False)
                
                # *target_21* updates
                
                # if target_21 is starting this frame...
                if target_21.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                    # keep track of start time/frame for later
                    target_21.frameNStart = frameN  # exact frame index
                    target_21.tStart = t  # local t and not account for scr refresh
                    target_21.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_21, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_21.started')
                    # update status
                    target_21.status = STARTED
                    target_21.setAutoDraw(True)
                
                # if target_21 is active this frame...
                if target_21.status == STARTED:
                    # update params
                    pass
                
                # if target_21 is stopping this frame...
                if target_21.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_21.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_21.tStop = t  # not accounting for scr refresh
                        target_21.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_21.stopped')
                        # update status
                        target_21.status = FINISHED
                        target_21.setAutoDraw(False)
                
                # *distractor_114* updates
                
                # if distractor_114 is starting this frame...
                if distractor_114.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_114.frameNStart = frameN  # exact frame index
                    distractor_114.tStart = t  # local t and not account for scr refresh
                    distractor_114.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_114, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_114.status = STARTED
                    distractor_114.setAutoDraw(True)
                
                # if distractor_114 is active this frame...
                if distractor_114.status == STARTED:
                    # update params
                    pass
                
                # if distractor_114 is stopping this frame...
                if distractor_114.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_114.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_114.tStop = t  # not accounting for scr refresh
                        distractor_114.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_114.status = FINISHED
                        distractor_114.setAutoDraw(False)
                
                # *distractor_132* updates
                
                # if distractor_132 is starting this frame...
                if distractor_132.status == NOT_STARTED and tThisFlip >= 0.636-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_132.frameNStart = frameN  # exact frame index
                    distractor_132.tStart = t  # local t and not account for scr refresh
                    distractor_132.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_132, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_132.status = STARTED
                    distractor_132.setAutoDraw(True)
                
                # if distractor_132 is active this frame...
                if distractor_132.status == STARTED:
                    # update params
                    pass
                
                # if distractor_132 is stopping this frame...
                if distractor_132.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_132.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_132.tStop = t  # not accounting for scr refresh
                        distractor_132.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_132.status = FINISHED
                        distractor_132.setAutoDraw(False)
                
                # *distractor_133* updates
                
                # if distractor_133 is starting this frame...
                if distractor_133.status == NOT_STARTED and tThisFlip >= 0.689-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_133.frameNStart = frameN  # exact frame index
                    distractor_133.tStart = t  # local t and not account for scr refresh
                    distractor_133.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_133, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_133.status = STARTED
                    distractor_133.setAutoDraw(True)
                
                # if distractor_133 is active this frame...
                if distractor_133.status == STARTED:
                    # update params
                    pass
                
                # if distractor_133 is stopping this frame...
                if distractor_133.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_133.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_133.tStop = t  # not accounting for scr refresh
                        distractor_133.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_133.status = FINISHED
                        distractor_133.setAutoDraw(False)
                
                # *distractor_134* updates
                
                # if distractor_134 is starting this frame...
                if distractor_134.status == NOT_STARTED and tThisFlip >= 0.742-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_134.frameNStart = frameN  # exact frame index
                    distractor_134.tStart = t  # local t and not account for scr refresh
                    distractor_134.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_134, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_134.status = STARTED
                    distractor_134.setAutoDraw(True)
                
                # if distractor_134 is active this frame...
                if distractor_134.status == STARTED:
                    # update params
                    pass
                
                # if distractor_134 is stopping this frame...
                if distractor_134.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_134.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_134.tStop = t  # not accounting for scr refresh
                        distractor_134.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_134.status = FINISHED
                        distractor_134.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_4Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_4" ---
            for thisComponent in routine_4Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('routine_4.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.795000)
            
            # --- Prepare to start Routine "response_400ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('response_400ms.started', globalClock.getTime())
            textbox_9.reset()
            textbox_10.reset()
            # reset button_5 to account for continued clicks & clear times on/off
            button_5.reset()
            # keep track of which components have finished
            response_400msComponents = [textbox_9, practice_response_instructions_5, textbox_10, button_5]
            for thisComponent in response_400msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response_400ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textbox_9* updates
                
                # if textbox_9 is starting this frame...
                if textbox_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_9.frameNStart = frameN  # exact frame index
                    textbox_9.tStart = t  # local t and not account for scr refresh
                    textbox_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_9.started')
                    # update status
                    textbox_9.status = STARTED
                    textbox_9.setAutoDraw(True)
                
                # if textbox_9 is active this frame...
                if textbox_9.status == STARTED:
                    # update params
                    pass
                
                # *practice_response_instructions_5* updates
                
                # if practice_response_instructions_5 is starting this frame...
                if practice_response_instructions_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    practice_response_instructions_5.frameNStart = frameN  # exact frame index
                    practice_response_instructions_5.tStart = t  # local t and not account for scr refresh
                    practice_response_instructions_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(practice_response_instructions_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_response_instructions_5.started')
                    # update status
                    practice_response_instructions_5.status = STARTED
                    practice_response_instructions_5.setAutoDraw(True)
                
                # if practice_response_instructions_5 is active this frame...
                if practice_response_instructions_5.status == STARTED:
                    # update params
                    pass
                
                # *textbox_10* updates
                
                # if textbox_10 is starting this frame...
                if textbox_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_10.frameNStart = frameN  # exact frame index
                    textbox_10.tStart = t  # local t and not account for scr refresh
                    textbox_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_10, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_10.started')
                    # update status
                    textbox_10.status = STARTED
                    textbox_10.setAutoDraw(True)
                
                # if textbox_10 is active this frame...
                if textbox_10.status == STARTED:
                    # update params
                    pass
                # *button_5* updates
                
                # if button_5 is starting this frame...
                if button_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_5.frameNStart = frameN  # exact frame index
                    button_5.tStart = t  # local t and not account for scr refresh
                    button_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_5.started')
                    # update status
                    button_5.status = STARTED
                    button_5.setAutoDraw(True)
                
                # if button_5 is active this frame...
                if button_5.status == STARTED:
                    # update params
                    pass
                    # check whether button_5 has been pressed
                    if button_5.isClicked:
                        if not button_5.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_5.timesOn.append(button_5.buttonClock.getTime())
                            button_5.timesOff.append(button_5.buttonClock.getTime())
                        elif len(button_5.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_5.timesOff[-1] = button_5.buttonClock.getTime()
                        if not button_5.wasClicked:
                            # end routine when button_5 is clicked
                            continueRoutine = False
                        if not button_5.wasClicked:
                            # run callback code when button_5 is clicked
                            pass
                # take note of whether button_5 was clicked, so that next frame we know if clicks are new
                button_5.wasClicked = button_5.isClicked and button_5.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in response_400msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response_400ms" ---
            for thisComponent in response_400msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('response_400ms.stopped', globalClock.getTime())
            # Run 'End Routine' code from accuracy_check_5
            if textbox_7.text == target1:
                thisExp.addData("f1_accuracy", 1)
            else:
                thisExp.addData("f1_accuracy", 0)
            
            if textbox_8.text == target2:
                thisExp.addData("f2_accuracy", 1)
            else:
                thisExp.addData("f2_accuracy", 0)
            
            
            trials_400ms.addData('textbox_9.text',textbox_9.text)
            trials_400ms.addData('textbox_10.text',textbox_10.text)
            trials_400ms.addData('button_5.numClicks', button_5.numClicks)
            if button_5.numClicks:
               trials_400ms.addData('button_5.timesOn', button_5.timesOn)
               trials_400ms.addData('button_5.timesOff', button_5.timesOff)
            else:
               trials_400ms.addData('button_5.timesOn', "")
               trials_400ms.addData('button_5.timesOff', "")
            # the Routine "response_400ms" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_400ms'
        
        
        # set up handler to look after randomisation of conditions etc
        trials_600ms = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('600ms.xlsx'),
            seed=None, name='trials_600ms')
        thisExp.addLoop(trials_600ms)  # add the loop to the experiment
        thisTrials_600m = trials_600ms.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_600m.rgb)
        if thisTrials_600m != None:
            for paramName in thisTrials_600m:
                globals()[paramName] = thisTrials_600m[paramName]
        
        for thisTrials_600m in trials_600ms:
            currentLoop = trials_600ms
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_600m.rgb)
            if thisTrials_600m != None:
                for paramName in thisTrials_600m:
                    globals()[paramName] = thisTrials_600m[paramName]
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # keep track of which components have finished
            fixationComponents = [cross]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross* updates
                
                # if cross is starting this frame...
                if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross.frameNStart = frameN  # exact frame index
                    cross.tStart = t  # local t and not account for scr refresh
                    cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.started')
                    # update status
                    cross.status = STARTED
                    cross.setAutoDraw(True)
                
                # if cross is active this frame...
                if cross.status == STARTED:
                    # update params
                    pass
                
                # if cross is stopping this frame...
                if cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        # update status
                        cross.status = FINISHED
                        cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "routine_5" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('routine_5.started', globalClock.getTime())
            target_22.setText(target1)
            target_23.setText(target2)
            # keep track of which components have finished
            routine_5Components = [distractor_115, target_22, distractor_116, distractor_117, distractor_118, distractor_119, distractor_120, distractor_121, distractor_122, distractor_123, distractor_124, distractor_125, distractor_126, distractor_127, target_23, distractor_128, distractor_129, distractor_130, distractor_131]
            for thisComponent in routine_5Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "routine_5" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.007:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *distractor_115* updates
                
                # if distractor_115 is starting this frame...
                if distractor_115.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_115.frameNStart = frameN  # exact frame index
                    distractor_115.tStart = t  # local t and not account for scr refresh
                    distractor_115.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_115, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_115.status = STARTED
                    distractor_115.setAutoDraw(True)
                
                # if distractor_115 is active this frame...
                if distractor_115.status == STARTED:
                    # update params
                    pass
                
                # if distractor_115 is stopping this frame...
                if distractor_115.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_115.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_115.tStop = t  # not accounting for scr refresh
                        distractor_115.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_115.status = FINISHED
                        distractor_115.setAutoDraw(False)
                
                # *target_22* updates
                
                # if target_22 is starting this frame...
                if target_22.status == NOT_STARTED and tThisFlip >= 0.053-frameTolerance:
                    # keep track of start time/frame for later
                    target_22.frameNStart = frameN  # exact frame index
                    target_22.tStart = t  # local t and not account for scr refresh
                    target_22.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_22, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_22.started')
                    # update status
                    target_22.status = STARTED
                    target_22.setAutoDraw(True)
                
                # if target_22 is active this frame...
                if target_22.status == STARTED:
                    # update params
                    pass
                
                # if target_22 is stopping this frame...
                if target_22.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_22.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_22.tStop = t  # not accounting for scr refresh
                        target_22.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_22.stopped')
                        # update status
                        target_22.status = FINISHED
                        target_22.setAutoDraw(False)
                
                # *distractor_116* updates
                
                # if distractor_116 is starting this frame...
                if distractor_116.status == NOT_STARTED and tThisFlip >= 0.105-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_116.frameNStart = frameN  # exact frame index
                    distractor_116.tStart = t  # local t and not account for scr refresh
                    distractor_116.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_116, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_116.status = STARTED
                    distractor_116.setAutoDraw(True)
                
                # if distractor_116 is active this frame...
                if distractor_116.status == STARTED:
                    # update params
                    pass
                
                # if distractor_116 is stopping this frame...
                if distractor_116.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_116.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_116.tStop = t  # not accounting for scr refresh
                        distractor_116.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_116.status = FINISHED
                        distractor_116.setAutoDraw(False)
                
                # *distractor_117* updates
                
                # if distractor_117 is starting this frame...
                if distractor_117.status == NOT_STARTED and tThisFlip >= 0.159-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_117.frameNStart = frameN  # exact frame index
                    distractor_117.tStart = t  # local t and not account for scr refresh
                    distractor_117.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_117, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_117.status = STARTED
                    distractor_117.setAutoDraw(True)
                
                # if distractor_117 is active this frame...
                if distractor_117.status == STARTED:
                    # update params
                    pass
                
                # if distractor_117 is stopping this frame...
                if distractor_117.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_117.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_117.tStop = t  # not accounting for scr refresh
                        distractor_117.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_117.status = FINISHED
                        distractor_117.setAutoDraw(False)
                
                # *distractor_118* updates
                
                # if distractor_118 is starting this frame...
                if distractor_118.status == NOT_STARTED and tThisFlip >= 0.212-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_118.frameNStart = frameN  # exact frame index
                    distractor_118.tStart = t  # local t and not account for scr refresh
                    distractor_118.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_118, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_118.status = STARTED
                    distractor_118.setAutoDraw(True)
                
                # if distractor_118 is active this frame...
                if distractor_118.status == STARTED:
                    # update params
                    pass
                
                # if distractor_118 is stopping this frame...
                if distractor_118.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_118.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_118.tStop = t  # not accounting for scr refresh
                        distractor_118.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_118.status = FINISHED
                        distractor_118.setAutoDraw(False)
                
                # *distractor_119* updates
                
                # if distractor_119 is starting this frame...
                if distractor_119.status == NOT_STARTED and tThisFlip >= 0.265-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_119.frameNStart = frameN  # exact frame index
                    distractor_119.tStart = t  # local t and not account for scr refresh
                    distractor_119.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_119, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_119.status = STARTED
                    distractor_119.setAutoDraw(True)
                
                # if distractor_119 is active this frame...
                if distractor_119.status == STARTED:
                    # update params
                    pass
                
                # if distractor_119 is stopping this frame...
                if distractor_119.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_119.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_119.tStop = t  # not accounting for scr refresh
                        distractor_119.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_119.status = FINISHED
                        distractor_119.setAutoDraw(False)
                
                # *distractor_120* updates
                
                # if distractor_120 is starting this frame...
                if distractor_120.status == NOT_STARTED and tThisFlip >= 0.318-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_120.frameNStart = frameN  # exact frame index
                    distractor_120.tStart = t  # local t and not account for scr refresh
                    distractor_120.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_120, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_120.status = STARTED
                    distractor_120.setAutoDraw(True)
                
                # if distractor_120 is active this frame...
                if distractor_120.status == STARTED:
                    # update params
                    pass
                
                # if distractor_120 is stopping this frame...
                if distractor_120.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_120.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_120.tStop = t  # not accounting for scr refresh
                        distractor_120.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_120.status = FINISHED
                        distractor_120.setAutoDraw(False)
                
                # *distractor_121* updates
                
                # if distractor_121 is starting this frame...
                if distractor_121.status == NOT_STARTED and tThisFlip >= 0.371-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_121.frameNStart = frameN  # exact frame index
                    distractor_121.tStart = t  # local t and not account for scr refresh
                    distractor_121.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_121, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_121.status = STARTED
                    distractor_121.setAutoDraw(True)
                
                # if distractor_121 is active this frame...
                if distractor_121.status == STARTED:
                    # update params
                    pass
                
                # if distractor_121 is stopping this frame...
                if distractor_121.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_121.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_121.tStop = t  # not accounting for scr refresh
                        distractor_121.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_121.status = FINISHED
                        distractor_121.setAutoDraw(False)
                
                # *distractor_122* updates
                
                # if distractor_122 is starting this frame...
                if distractor_122.status == NOT_STARTED and tThisFlip >= 0.424-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_122.frameNStart = frameN  # exact frame index
                    distractor_122.tStart = t  # local t and not account for scr refresh
                    distractor_122.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_122, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_122.status = STARTED
                    distractor_122.setAutoDraw(True)
                
                # if distractor_122 is active this frame...
                if distractor_122.status == STARTED:
                    # update params
                    pass
                
                # if distractor_122 is stopping this frame...
                if distractor_122.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_122.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_122.tStop = t  # not accounting for scr refresh
                        distractor_122.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_122.status = FINISHED
                        distractor_122.setAutoDraw(False)
                
                # *distractor_123* updates
                
                # if distractor_123 is starting this frame...
                if distractor_123.status == NOT_STARTED and tThisFlip >= 0.477-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_123.frameNStart = frameN  # exact frame index
                    distractor_123.tStart = t  # local t and not account for scr refresh
                    distractor_123.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_123, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_123.status = STARTED
                    distractor_123.setAutoDraw(True)
                
                # if distractor_123 is active this frame...
                if distractor_123.status == STARTED:
                    # update params
                    pass
                
                # if distractor_123 is stopping this frame...
                if distractor_123.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_123.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_123.tStop = t  # not accounting for scr refresh
                        distractor_123.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_123.status = FINISHED
                        distractor_123.setAutoDraw(False)
                
                # *distractor_124* updates
                
                # if distractor_124 is starting this frame...
                if distractor_124.status == NOT_STARTED and tThisFlip >= 0.53-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_124.frameNStart = frameN  # exact frame index
                    distractor_124.tStart = t  # local t and not account for scr refresh
                    distractor_124.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_124, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_124.status = STARTED
                    distractor_124.setAutoDraw(True)
                
                # if distractor_124 is active this frame...
                if distractor_124.status == STARTED:
                    # update params
                    pass
                
                # if distractor_124 is stopping this frame...
                if distractor_124.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_124.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_124.tStop = t  # not accounting for scr refresh
                        distractor_124.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_124.status = FINISHED
                        distractor_124.setAutoDraw(False)
                
                # *distractor_125* updates
                
                # if distractor_125 is starting this frame...
                if distractor_125.status == NOT_STARTED and tThisFlip >= 0.583-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_125.frameNStart = frameN  # exact frame index
                    distractor_125.tStart = t  # local t and not account for scr refresh
                    distractor_125.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_125, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_125.status = STARTED
                    distractor_125.setAutoDraw(True)
                
                # if distractor_125 is active this frame...
                if distractor_125.status == STARTED:
                    # update params
                    pass
                
                # if distractor_125 is stopping this frame...
                if distractor_125.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_125.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_125.tStop = t  # not accounting for scr refresh
                        distractor_125.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_125.status = FINISHED
                        distractor_125.setAutoDraw(False)
                
                # *distractor_126* updates
                
                # if distractor_126 is starting this frame...
                if distractor_126.status == NOT_STARTED and tThisFlip >= 0.636-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_126.frameNStart = frameN  # exact frame index
                    distractor_126.tStart = t  # local t and not account for scr refresh
                    distractor_126.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_126, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_126.status = STARTED
                    distractor_126.setAutoDraw(True)
                
                # if distractor_126 is active this frame...
                if distractor_126.status == STARTED:
                    # update params
                    pass
                
                # if distractor_126 is stopping this frame...
                if distractor_126.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_126.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_126.tStop = t  # not accounting for scr refresh
                        distractor_126.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_126.status = FINISHED
                        distractor_126.setAutoDraw(False)
                
                # *distractor_127* updates
                
                # if distractor_127 is starting this frame...
                if distractor_127.status == NOT_STARTED and tThisFlip >= 0.689-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_127.frameNStart = frameN  # exact frame index
                    distractor_127.tStart = t  # local t and not account for scr refresh
                    distractor_127.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_127, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_127.status = STARTED
                    distractor_127.setAutoDraw(True)
                
                # if distractor_127 is active this frame...
                if distractor_127.status == STARTED:
                    # update params
                    pass
                
                # if distractor_127 is stopping this frame...
                if distractor_127.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_127.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_127.tStop = t  # not accounting for scr refresh
                        distractor_127.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_127.status = FINISHED
                        distractor_127.setAutoDraw(False)
                
                # *target_23* updates
                
                # if target_23 is starting this frame...
                if target_23.status == NOT_STARTED and tThisFlip >= 0.742-frameTolerance:
                    # keep track of start time/frame for later
                    target_23.frameNStart = frameN  # exact frame index
                    target_23.tStart = t  # local t and not account for scr refresh
                    target_23.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_23, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_23.started')
                    # update status
                    target_23.status = STARTED
                    target_23.setAutoDraw(True)
                
                # if target_23 is active this frame...
                if target_23.status == STARTED:
                    # update params
                    pass
                
                # if target_23 is stopping this frame...
                if target_23.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > target_23.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        target_23.tStop = t  # not accounting for scr refresh
                        target_23.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_23.stopped')
                        # update status
                        target_23.status = FINISHED
                        target_23.setAutoDraw(False)
                
                # *distractor_128* updates
                
                # if distractor_128 is starting this frame...
                if distractor_128.status == NOT_STARTED and tThisFlip >= 0.795-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_128.frameNStart = frameN  # exact frame index
                    distractor_128.tStart = t  # local t and not account for scr refresh
                    distractor_128.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_128, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_128.status = STARTED
                    distractor_128.setAutoDraw(True)
                
                # if distractor_128 is active this frame...
                if distractor_128.status == STARTED:
                    # update params
                    pass
                
                # if distractor_128 is stopping this frame...
                if distractor_128.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_128.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_128.tStop = t  # not accounting for scr refresh
                        distractor_128.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_128.status = FINISHED
                        distractor_128.setAutoDraw(False)
                
                # *distractor_129* updates
                
                # if distractor_129 is starting this frame...
                if distractor_129.status == NOT_STARTED and tThisFlip >= 0.848-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_129.frameNStart = frameN  # exact frame index
                    distractor_129.tStart = t  # local t and not account for scr refresh
                    distractor_129.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_129, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_129.status = STARTED
                    distractor_129.setAutoDraw(True)
                
                # if distractor_129 is active this frame...
                if distractor_129.status == STARTED:
                    # update params
                    pass
                
                # if distractor_129 is stopping this frame...
                if distractor_129.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_129.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_129.tStop = t  # not accounting for scr refresh
                        distractor_129.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_129.status = FINISHED
                        distractor_129.setAutoDraw(False)
                
                # *distractor_130* updates
                
                # if distractor_130 is starting this frame...
                if distractor_130.status == NOT_STARTED and tThisFlip >= 0.901-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_130.frameNStart = frameN  # exact frame index
                    distractor_130.tStart = t  # local t and not account for scr refresh
                    distractor_130.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_130, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_130.status = STARTED
                    distractor_130.setAutoDraw(True)
                
                # if distractor_130 is active this frame...
                if distractor_130.status == STARTED:
                    # update params
                    pass
                
                # if distractor_130 is stopping this frame...
                if distractor_130.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_130.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_130.tStop = t  # not accounting for scr refresh
                        distractor_130.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_130.status = FINISHED
                        distractor_130.setAutoDraw(False)
                
                # *distractor_131* updates
                
                # if distractor_131 is starting this frame...
                if distractor_131.status == NOT_STARTED and tThisFlip >= 0.954-frameTolerance:
                    # keep track of start time/frame for later
                    distractor_131.frameNStart = frameN  # exact frame index
                    distractor_131.tStart = t  # local t and not account for scr refresh
                    distractor_131.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(distractor_131, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    distractor_131.status = STARTED
                    distractor_131.setAutoDraw(True)
                
                # if distractor_131 is active this frame...
                if distractor_131.status == STARTED:
                    # update params
                    pass
                
                # if distractor_131 is stopping this frame...
                if distractor_131.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > distractor_131.tStartRefresh + 0.053-frameTolerance:
                        # keep track of stop time/frame for later
                        distractor_131.tStop = t  # not accounting for scr refresh
                        distractor_131.frameNStop = frameN  # exact frame index
                        # update status
                        distractor_131.status = FINISHED
                        distractor_131.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_5Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_5" ---
            for thisComponent in routine_5Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('routine_5.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.007000)
            
            # --- Prepare to start Routine "response_600ms" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('response_600ms.started', globalClock.getTime())
            textbox_11.reset()
            textbox_12.reset()
            # reset button_6 to account for continued clicks & clear times on/off
            button_6.reset()
            # keep track of which components have finished
            response_600msComponents = [textbox_11, practice_response_instructions_6, textbox_12, button_6]
            for thisComponent in response_600msComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response_600ms" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textbox_11* updates
                
                # if textbox_11 is starting this frame...
                if textbox_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_11.frameNStart = frameN  # exact frame index
                    textbox_11.tStart = t  # local t and not account for scr refresh
                    textbox_11.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_11, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_11.started')
                    # update status
                    textbox_11.status = STARTED
                    textbox_11.setAutoDraw(True)
                
                # if textbox_11 is active this frame...
                if textbox_11.status == STARTED:
                    # update params
                    pass
                
                # *practice_response_instructions_6* updates
                
                # if practice_response_instructions_6 is starting this frame...
                if practice_response_instructions_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    practice_response_instructions_6.frameNStart = frameN  # exact frame index
                    practice_response_instructions_6.tStart = t  # local t and not account for scr refresh
                    practice_response_instructions_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(practice_response_instructions_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_response_instructions_6.started')
                    # update status
                    practice_response_instructions_6.status = STARTED
                    practice_response_instructions_6.setAutoDraw(True)
                
                # if practice_response_instructions_6 is active this frame...
                if practice_response_instructions_6.status == STARTED:
                    # update params
                    pass
                
                # *textbox_12* updates
                
                # if textbox_12 is starting this frame...
                if textbox_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_12.frameNStart = frameN  # exact frame index
                    textbox_12.tStart = t  # local t and not account for scr refresh
                    textbox_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_12, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_12.started')
                    # update status
                    textbox_12.status = STARTED
                    textbox_12.setAutoDraw(True)
                
                # if textbox_12 is active this frame...
                if textbox_12.status == STARTED:
                    # update params
                    pass
                # *button_6* updates
                
                # if button_6 is starting this frame...
                if button_6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_6.frameNStart = frameN  # exact frame index
                    button_6.tStart = t  # local t and not account for scr refresh
                    button_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_6.started')
                    # update status
                    button_6.status = STARTED
                    button_6.setAutoDraw(True)
                
                # if button_6 is active this frame...
                if button_6.status == STARTED:
                    # update params
                    pass
                    # check whether button_6 has been pressed
                    if button_6.isClicked:
                        if not button_6.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_6.timesOn.append(button_6.buttonClock.getTime())
                            button_6.timesOff.append(button_6.buttonClock.getTime())
                        elif len(button_6.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_6.timesOff[-1] = button_6.buttonClock.getTime()
                        if not button_6.wasClicked:
                            # end routine when button_6 is clicked
                            continueRoutine = False
                        if not button_6.wasClicked:
                            # run callback code when button_6 is clicked
                            pass
                # take note of whether button_6 was clicked, so that next frame we know if clicks are new
                button_6.wasClicked = button_6.isClicked and button_6.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in response_600msComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response_600ms" ---
            for thisComponent in response_600msComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('response_600ms.stopped', globalClock.getTime())
            # Run 'End Routine' code from accuracy_check_6
            if textbox_11.text == target1:
                thisExp.addData("f1_accuracy", 1)
            else:
                thisExp.addData("f1_accuracy", 0)
            
            if textbox_12.text == target2:
                thisExp.addData("f2_accuracy", 1)
            else:
                thisExp.addData("f2_accuracy", 0)
            
            
            trials_600ms.addData('textbox_11.text',textbox_11.text)
            trials_600ms.addData('textbox_12.text',textbox_12.text)
            trials_600ms.addData('button_6.numClicks', button_6.numClicks)
            if button_6.numClicks:
               trials_600ms.addData('button_6.timesOn', button_6.timesOn)
               trials_600ms.addData('button_6.timesOff', button_6.timesOff)
            else:
               trials_600ms.addData('button_6.timesOn', "")
               trials_600ms.addData('button_6.timesOff', "")
            # the Routine "response_600ms" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_600ms'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'counter_balance'
    
    
    # --- Prepare to start Routine "thank_you" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thank_you.started', globalClock.getTime())
    key_instruct_4.keys = []
    key_instruct_4.rt = []
    _key_instruct_4_allKeys = []
    # keep track of which components have finished
    thank_youComponents = [text_norm_3, key_instruct_4]
    for thisComponent in thank_youComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_3* updates
        
        # if text_norm_3 is starting this frame...
        if text_norm_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_3.frameNStart = frameN  # exact frame index
            text_norm_3.tStart = t  # local t and not account for scr refresh
            text_norm_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_3.status = STARTED
            text_norm_3.setAutoDraw(True)
        
        # if text_norm_3 is active this frame...
        if text_norm_3.status == STARTED:
            # update params
            pass
        
        # *key_instruct_4* updates
        waitOnFlip = False
        
        # if key_instruct_4 is starting this frame...
        if key_instruct_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_4.frameNStart = frameN  # exact frame index
            key_instruct_4.tStart = t  # local t and not account for scr refresh
            key_instruct_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_4.started')
            # update status
            key_instruct_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_4.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_4_allKeys.extend(theseKeys)
            if len(_key_instruct_4_allKeys):
                key_instruct_4.keys = _key_instruct_4_allKeys[0].name  # just the first key pressed
                key_instruct_4.rt = _key_instruct_4_allKeys[0].rt
                key_instruct_4.duration = _key_instruct_4_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_youComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_youComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thank_you.stopped', globalClock.getTime())
    # check responses
    if key_instruct_4.keys in ['', [], None]:  # No response was made
        key_instruct_4.keys = None
    thisExp.addData('key_instruct_4.keys',key_instruct_4.keys)
    if key_instruct_4.keys != None:  # we had a response
        thisExp.addData('key_instruct_4.rt', key_instruct_4.rt)
        thisExp.addData('key_instruct_4.duration', key_instruct_4.duration)
    thisExp.nextEntry()
    # the Routine "thank_you" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
