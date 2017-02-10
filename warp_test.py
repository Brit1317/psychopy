#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from psychopy import visual, core
from psychopy.visual import stereo
from psychopy.visual import windowwarp

win = visual.Window(fullscr=True, color=(0,0,0), 
    size=(1920,1080), waitBlanking=False, useFBO=True)
print(win.getActualFrameRate(nIdentical=10, nMaxFrames=1000, nWarmUpFrames=120, threshold=1))

warper = windowwarp.Warper(win,
    warp='spherical',
    warpfile = "",
    warpGridsize = 128,
    eyepoint = [0.5, 0.5],
    flipHorizontal = False,
    flipVertical = False)

warper.dist_cm = 15
warper.changeProjection("spherical")

msg = visual.TextStim(win, text=u"THIS IS 3D", pos=(0,0), antialias=True)
l = visual.Line(win, start=(-0.5, -0.5), end=(0.5, 0.5), interpolate=True)

n = 0
while n < 10000:
    #msg.text="Left"
    #msg.pos=(0.01,0)
    #msg.draw()
    #l.end = (0.5,0.5)
    #l.draw()
    msg.text="Right"
    msg.pos=(-0.01,0)
    msg.draw()
    l.end = (0.51,0.5)
    l.draw()

    win.flip()

    n += 1

win.close()
core.quit()
