#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from psychopy import visual, core
from psychopy.visual import stereo

win = visual.stereo.SpannedWindow(fullscr=True, color=(0,0,0), 
    size=(1920,1080), waitBlanking=True)
print(win.getActualFrameRate(nIdentical=10, nMaxFrames=1000, nWarmUpFrames=120, threshold=1))

msg = visual.TextStim(win, text=u"THIS IS 3D", pos=(0,0), antialias=False)
l = visual.Line(win, start=(-0.5, -0.5), end=(0.5, 0.5), interpolate=True)

n = 0
while n < 1200:
    win.setBuffer('left')
    msg.text="Test"
    msg.pos=(0.01,0)
    msg.draw()
    l.end = (0.5,0.5)
    l.draw()
    win.setBuffer('right')
    msg.text="Test"
    msg.pos=(-0.01,0)
    msg.draw()
    l.end = (0.51,0.5)
    l.draw()

    win.flip()

    n += 1

win.close()
core.quit()
