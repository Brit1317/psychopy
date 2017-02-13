#!/usr/bin/env python2

"""
Stereo display wrappers and utilities. Written by Matthew Cutone (2016) with
code modified from visual.Window by PsychoPy contributers.
"""

# python stdlibs
import ctypes
import sys
import math

# third-party libs
import numpy
import pyglet
pyglet.options['debug_gl'] = False
GL = pyglet.gl

# PsychoPy imports
from . import shaders as _shaders
from . import window
from .. import platform_specific
from .. import logging
from . import globalVars
from . import windowwarp

reportNDroppedFrames = 5

class Framebuffer(object):
    """Class for generating and managing an off-screen render target.
    """

    def __init__(self, win, size=(800,600), bg_color=(0,0,0)):
        """Framebuffer Objects provide a means of rendering scenes off-screen
        to textures. These textures can be filtered by programmable GPU shaders
        and/or applied to quads for blitting and warping.

        Framebuffers generated from this class have a single color and render
        buffer by default. The color buffer has a single attachment at
        GL_COLOR_ATTACHMENT0_EXT with a GL_RGBA16 internal colour format. The
        texture can be accessed by its handle 'textureId'.
        """

        # window pointer
        self.win = win
        self.bg_color = bg_color
        self.fbo_size = numpy.array(size, numpy.int)

        # create the framebuffer, must succeed or crash
        self._init_framebuffer()

        # flag to indicate that drawing is in stereo

    def __repr__(self):
        """Return the framebuffer handle ID when called"""
        return self.frameBufferId

    def __del__(self):
        """Delete the framebuffer and attached resources"""
        # check if a color and render buffers have valid IDs (non-zero), if so,
        # delete them
        if self.frameBufferId:
            GL.glDeleteTextures(1, ctypes.byref(self.textureId))
        if self.renderBufferId:
            GL.glDeleteRenderbuffersEXT(1, ctypes.byref(self.renderBufferId))

        # discard the framebuffer to free up resources
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)
        GL.glDeleteFramebuffersEXT(1, ctypes.byref(self.frameBufferId))

    def _init_framebuffer(self):
        """Setup frambuffer object for off-screen rendering. Setup is much like
        that in the window.Window class."""

        # get a new framebuffer reference handle
        self.frameBufferId = GL.GLuint()
        GL.glGenFramebuffersEXT(1, ctypes.byref(self.frameBufferId))
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)

        # create a texture to behave as a render target
        self.textureId = GL.GLuint()
        GL.glGenTextures(1, ctypes.byref(self.textureId))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureId)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16, self.fbo_size[0],
                        self.fbo_size[1], 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glFramebufferTexture2DEXT(GL.GL_FRAMEBUFFER_EXT,
                                     GL.GL_COLOR_ATTACHMENT0_EXT,
                                     GL.GL_TEXTURE_2D, self.textureId, 0)

        # clear the colour attachment
        GL.glClearColor(self.bg_color[0],
                        self.bg_color[1],
                        self.bg_color[2],
                        0.0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # create and attach a renderbuffer
        self.renderBufferId = GL.GLuint()
        GL.glGenRenderbuffersEXT(1, ctypes.byref(
            self.renderBufferId))
        GL.glBindRenderbufferEXT(GL.GL_RENDERBUFFER_EXT, self.renderBufferId)
        GL.glRenderbufferStorageEXT(GL.GL_RENDERBUFFER_EXT,
                                    GL.GL_DEPTH24_STENCIL8_EXT,
                                    self.fbo_size[0], self.fbo_size[1])
        GL.glFramebufferRenderbufferEXT(GL.GL_FRAMEBUFFER_EXT,
                                        GL.GL_STENCIL_ATTACHMENT_EXT,
                                        GL.GL_RENDERBUFFER_EXT,
                                        self.renderBufferId)

        # clear depth and stencil buffer
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        # set the default projection
        self.reset_projection()

        # check if the framebuffer has been successfully initalized
        status = GL.glCheckFramebufferStatusEXT(GL.GL_FRAMEBUFFER_EXT)
        if status != GL.GL_FRAMEBUFFER_COMPLETE_EXT:
            logging.error("Error in framebuffer activation")
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)
            print("FATAL: Framebuffer failed to initialize, your hardware" \
                  " or drivers may not support this feature.")
            #sys.exit(1) # exit more nicely than this

            return None

        GL.glDisable(GL.GL_TEXTURE_2D)
    
    def set_default_projection(self):
        """Set projection to defualt that stimuli classes expect"""
        # NB - this should be in the window class?
        GL.glViewport(0, 0, int(self.fbo_size[0]), int(self.fbo_size[1]))
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.gluOrtho2D(-1, 1, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
    
    def set_stereo_projection(self, eye_view='left', eye_sep=0.060, converge_dist=1.0, 
                              fovy=90.0, mode='offaxis', clip=(0.5, 50.0)):
        """Set the camera settings for a stereo projection. One FBO is used per view (eye image),
        adapted from http://www.orthostereo.com/geometryopengl.html for now"""
        # NB - this should be in the window class?
        # must be bound to the current FBO
        aspect_ratio = float(self.fbo_size[0]) / float(self.fbo_size[1])

        # convert FOV to radians
        fovy = fovy * (180 / math.pi)

        top = clip[0] * math.tan(fovy / 2.0)
        right = aspect_ratio * top
        offset = (clip[0] / float(converge_dist)) * (eye_sep / 2.0)

        frust_top = top
        frust_bottom = -top
        frust_left = -right - offset
        frust_right = right - offset

        if eye_view == 'left':
            view_shift = eye_sep / 2.0
        elif eye_view == 'right':
            view_shift = -eye_sep / 2.0

        # calculate the camera settings and set them for each mode
        if mode == 'offaxis' or mode == 'asymm':
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glFrustum(frust_left,
                          frust_right,
                          frust_bottom,
                          frust_top,
                          clip[0],
                          clip[1])
            GL.glTranslatef(view_shift, 0.0, 0.0)
            
            # move camera away from the screen plane
            GL.glTranslatef(0.0, 0.0, -converge_dist)

            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
        
        else:
            pass

    def bind_texture(self, where=GL.GL_TEXTURE0):
        """Bind the FBO texture to a given texture unit"""
        GL.glActiveTexture(where)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureId)
        GL.glColorMask(True, True, True, True)

    def unbind_texture(self, to_target=0):
        """Unbind the framebuffer's texture"""
        GL.glBindTexture(GL.GL_TEXTURE_2D, to_target)

    def bind_fbo(self):
        """Convienence function to bind FBO for read and draw"""
        # only simple texture being used
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)
        GL.glViewport(0, 0, self.fbo_size[0], self.fbo_size[1])
        # force the main window to report the FBO size, very hacky 
        # but needed for stimuli to draw correctly.
        self.win.size = self.fbo_size 

    def unbind_fbo(self, toTarget=0):
        """Unbind to default framebuffer to default (0)"""
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, toTarget)

    def clear_buffer(self, buffer=GL.GL_COLOR_BUFFER_BIT):
        """Clear the specified FBO attachment/buffer"""
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)
        GL.glClear(buffer)

class MultiRenderWindow(window.Window):

    """Class for rendering stereoscopic scenes using multiple Framebuffers.

    Main support class for stereo modes that require off screen rendering for
    each eye's image; intended to provide alternative stereo display modes and
    stereo support for systems that do not support the GL_STEREO extension.

    This class should only be instanced by stereo wrappers or when implimenting
    a custom stereo window class. Using quad-buffers for stereo output is not
    permitted through this class, instead use the stereo=True option in the
    default visual.Window class.

    Usage:

        You can set which eye is the render target by calling the setBuffer()
        method and specifying either 'left' or 'right'. Subsiquent draw
        commands will be rendered to that eye. This behaviour is consistent
        across all stereo modes. Below is an example of how this works.

            win.setBuffer('left')
            left_stim.draw()        # drawn only on left eye image
            win.setBuffer('right')
            right_stim.draw()       # drawn only on right eye image

        Images are presented using the shader supplied by the inheriting class
        at flip.

    Caveats:

    1.   Video driver support for Framebuffer objects (FBOs) and
         programmable fragment shaders is required (OpenGL 2.1+).
         Most modern graphics cards will have support for these functions.
         Ensure your graphics drivers are updated for best results.

    2.   Stimulus functions such as setAutoDraw() and mouse pointer hit checks
         behave unexpectedly when using stereo modes. It is best to call the
         draw() stimulus class method directly when rendering and writing your
         own click detection system.

    """

    def __init__(self, *args, **kwargs):

        # check for troublesome window settings
        if 'stereo' in kwargs:
            logging.warning("Requesting a quad-buffered stereo window is not "
                "allowed when using a 'MultiRenderWindow' sub-class. Disabling."
                )

            kwargs['stereo'] = False
        
        if 'useFBO' in kwargs:
            logging.warning("Using the built-in FBO support is not allowed "
            "when using MultiRenderWindow, disabling."
                )

            kwargs['useFBO'] = False
        
        # init window class
        super(MultiRenderWindow, self).__init__(*args, **kwargs)

        # a list of view buffers attached to this window, the stereo display type
        # determines what this contains
        self._view_buffers = list()

    def _endOfFlip(self, clearBuffer):
        """
        Clear both left and right eye framebuffer colour attachments if needed.
        """
        if clearBuffer:
            # clear the framebuffer color buffers
            self.rightFBO.clear_buffer()
            self.leftFBO.clear_buffer()

    def _prepareFBOrender(self):
        """Bind and configure the stereo shader. This is overridden by shaders
        needing more options."""
        GL.glUseProgram(self._progStereoShader)

    def _afterSetupGL(self):
        """After GL has been setup by the window, this function is called to
        setup the framebuffer to contain the off-screen windows for rendering
        stereo pairs.
        """
        # setup stereo frambuffers
        success = self._setupStereoFBO()
        if not success:
            logging.critical("Cannot create required FBO for stereo rendering."
                             " Exiting.")
            # todo: crash
        else:
            # compile the stereo shader included in the instancing class
            self._compileStereoShader()
            self._size = self.size # retain a copy of the fullscreen's size

    def _compileStereoShader(self):
        """Compile a stereo shader required for the given stereo mode, this is
        overridden by the inheriting class"""
        self._progStereoShader = self._progFBOtoFrame

    def _getFBOSize(self):
        """Some stereo modes need render targets with different size than
        the window. Override this method to supply a custom size."""
        return self._size
    
    def getMaxSamples(self):
        samples = (GL.GLint)()
        GL.glGetIntegerv(GL.GL_MAX_SAMPLES, samples)
        
        return samples.value

    def _setupStereoFBO(self):
        """Setup a Frame Buffer Objects to handle offscreen rendering of eye
        images.

        Reserves the last few color attachments for offscreen drawing. Most
        modern cards support at least 8 of them so the rest are free for other
        purposes.

        Each texture gets it's own FBO. This avoids the FBO re-validation the
        driver must do when switching attachments, leading to increased overhead
        and latency.
        """

        # init framebuffer objects as render targets
        self.leftFBO = Framebuffer(self, (self.size[0], self.size[1]))
        self.rightFBO = Framebuffer(self, (self.size[0], self.size[1]))

        return True

    def flip(self, clearBuffer=True):
        """
        * Copy of window.Window.flip() method with some modifications to better
        utilize the stereo rendering pipeline without the risk of breaking
        existing experiments. useFBO is prohibited by the stereo extension
        and is not tested for or applied here. *

        Flip the front and back buffers after drawing everything for your
        frame. (This replaces the win.update() method, better reflecting what
        is happening underneath).

        If clearBuffer=True, the left and right eye framebuffers are cleared.
        The result of the stereo shaders completely overwrites the pixels in the
        back buffer so it does not need to be cleared.
        """

        # NB - This is broken when using stereo displays
        for thisStim in self._toDraw:
            thisStim.draw()

        flipThisFrame = self._startOfFlip()
        if flipThisFrame:
            self._prepareFBOrender()
            # need blit the frambuffer object to the actual back buffer

            # unbind the framebuffer as the render target
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)
            GL.glDisable(GL.GL_BLEND)
            stencilOn = GL.glIsEnabled(GL.GL_STENCIL_TEST)
            GL.glDisable(GL.GL_STENCIL_TEST)

            if self.bits != None:
                self.bits._prepareFBOrender()
            
            GL.glViewport(0, 0, self._size[0], self._size[1])
            self._stereoRender() # call stereo rendering function   

            GL.glEnable(GL.GL_BLEND)
            self._finishFBOrender()

        # call this before flip() whether FBO was used or not
        self._afterFBOrender()

        if self.winType == "pyglet":
            # make sure this is current context
            if globalVars.currWindow != self:
                self.winHandle.switch_to()
                globalVars.currWindow = self

            GL.glTranslatef(0.0, 0.0, -5.0)

            for dispatcher in self._eventDispatchers:
                dispatcher.dispatch_events()

            # this might need to be done even more often than once per frame?
            self.winHandle.dispatch_events()

            # for pyglet 1.1.4 you needed to call media.dispatch for
            # movie updating
            if pyglet.version < '1.2':
                pyglet.media.dispatch_events()  # for sounds to be processed
            if flipThisFrame:
                self.winHandle.flip()
        else:
            if pygame.display.get_init():
                if flipThisFrame:
                    pygame.display.flip()
                # keeps us in synch with system event queue
                pygame.event.pump()
            else:
                core.quit()  # we've unitialised pygame so quit

        if flipThisFrame:
            # set to no active rendering texture
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            if stencilOn:
                GL.glEnable(GL.GL_STENCIL_TEST)
            #GL.glEnable(GL.GL_MULTISAMPLE)

        # rescale, reposition, & rotate
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        if self.viewScale is not None:
            GL.glScalef(self.viewScale[0], self.viewScale[1], 1)
            absScaleX = abs(self.viewScale[0])
            absScaleY = abs(self.viewScale[1])
        else:
            absScaleX, absScaleY = 1, 1

        if self.viewPos is not None:
            normRfPosX = self.viewPos[0] / absScaleX
            normRfPosY = self.viewPos[1] / absScaleY
            GL.glTranslatef(normRfPosX, normRfPosY, 0.0)

        if self.viewOri:  # float
            # the logic below for flip is partially correct, but does not
            # handle a nonzero viewPos
            flip = 1
            if self.viewScale is not None:
                _f = self.viewScale[0] * self.viewScale[1]
                if _f < 0:
                    flip = -1
            GL.glRotatef(flip * self.viewOri, 0.0, 0.0, -1.0)

        # reset returned buffer for next frame
        self._endOfFlip(clearBuffer)

        # waitBlanking
        if self.waitBlanking and flipThisFrame:
            GL.glBegin(GL.GL_POINTS)
            GL.glColor4f(0, 0, 0, 0)
            if sys.platform == 'win32' and self.glVendor.startswith('ati'):
                pass
            else:
                # this corrupts text rendering on win with some ATI cards :-(
                GL.glVertex2i(10, 10)
            GL.glEnd()
            GL.glFinish()

        # get timestamp
        now = logging.defaultClock.getTime()

        # run other functions immediately after flip completes
        for callEntry in self._toCall:
            callEntry['function'](*callEntry['args'], **callEntry['kwargs'])
        del self._toCall[:]

        # do bookkeeping
        if self.recordFrameIntervals:
            self.frames += 1
            deltaT = now - self.lastFrameT
            self.lastFrameT = now
            if self.recordFrameIntervalsJustTurnedOn:  # don't do anything
                self.recordFrameIntervalsJustTurnedOn = False
            else:  # past the first frame since turned on
                self.frameIntervals.append(deltaT)
                if deltaT > self._refreshThreshold:
                    self.nDroppedFrames += 1
                    if self.nDroppedFrames < reportNDroppedFrames:
                        txt = 't of last frame was %.2fms (=1/%i)'
                        msg = txt % (deltaT * 1000, 1 / deltaT)
                        logging.warning(msg, t=now)
                    elif self.nDroppedFrames == reportNDroppedFrames:
                        logging.warning("Multiple dropped frames have "
                                        "occurred - I'll stop bothering you "
                                        "about them!")

        # log events
        for logEntry in self._toLog:
            # {'msg':msg, 'level':level, 'obj':copy.copy(obj)}
            logging.log(msg=logEntry['msg'],
                        level=logEntry['level'],
                        t=now,
                        obj=logEntry['obj'])
        del self._toLog[:]

        # keep the system awake (prevent screen-saver or sleep)
        platform_specific.sendStayAwake()

        #    If self.waitBlanking is True, then return the time that
        # GL.glFinish() returned, set as the 'now' variable. Otherwise
        # return None as before
        #
        if self.waitBlanking is True:
            return now

    def setBuffer(self, buffer='left', clear=False):
        """
        Similar to setBuffer in Window, but changes the active eye FBO
        instead of the back quad-buffer. Buffers can be cleared individually
        if needed.
        """
        if buffer == 'left':
            self.leftFBO.bind_fbo()
            self.leftFBO.set_stereo_projection(eye_view='right')

        elif buffer == 'right':
            self.rightFBO.bind_fbo()
            self.leftFBO.set_stereo_projection(eye_view='right')

        if clear:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    
    def _finishFBOrender(self):
        GL.glUseProgram(0)

    def _stereoRender(self):
        """Override with code to configure and execute transfering stereo
        buffers to textures and blitting them to the back buffer."""
        pass

"""
Below are the stereo window classes that produce various stereo displays. Each
one requires FBO support to function therefore are limited to OpenGL 2.1+. For
quad-buffer enabled through the GL_STEREO extension, use the stereo=True
option included in the vanilla Window class.
"""

class PackedWindow(MultiRenderWindow):

    """Create a display where left and right eye images are packed horizontally
    or vertically.

    Set 'packing' to 'horizontal' (Left-Right) or 'vertical' (Top-Bottom),
    default is horizontal. 
    """

    def __init__(self, *args, **kwargs):
        
        self.packing = kwargs.pop('packing', 'horizontal')

        if self.packing not in ('horizontal', 'vertical'):
            # error, not valid packing mode specified
            logging.warning('Requested packing mode {} is invalid, defaulting'
                            'to horizontal')
            self.packing = 'horizontal'

        MultiRenderWindow.__init__(self, *args, **kwargs)

    def _renderLeftFBO(self):
        if self.packing == 'horizontal':
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 0.0)
            GL.glVertex2f(-1.0, -1.0)
            GL.glTexCoord2f(0.0, 1.0)
            GL.glVertex2f(-1.0, 1.0)
            GL.glTexCoord2f(1.0, 1.0)
            GL.glVertex2f(0.0, 1.0)
            GL.glTexCoord2f(1.0, 0.0)
            GL.glVertex2f(0.0, -1.0)
            GL.glEnd()
        elif self.packing == 'vertical':
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 0.0)
            GL.glVertex2f(-1.0, 0.0)
            GL.glTexCoord2f(0.0, 1.0)
            GL.glVertex2f(-1.0, 1.0)
            GL.glTexCoord2f(1.0, 1.0)
            GL.glVertex2f(1.0, 1.0)
            GL.glTexCoord2f(1.0, 0.0)
            GL.glVertex2f(1.0, 0.0)
            GL.glEnd()
    
    def _renderRightFBO(self):
        if self.packing == 'horizontal':
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 0.0)
            GL.glVertex2f(0.0, -1.0)
            GL.glTexCoord2f(0.0, 1.0)
            GL.glVertex2f(0.0, 1.0)
            GL.glTexCoord2f(1.0, 1.0)
            GL.glVertex2f(1.0, 1.0)
            GL.glTexCoord2f(1.0, 0.0)
            GL.glVertex2f(1.0, -1.0)
            GL.glEnd()
        elif self.packing == 'vertical':
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 0.0)
            GL.glVertex2f(-1.0, -1.0)
            GL.glTexCoord2f(0.0, 1.0)
            GL.glVertex2f(-1.0, 0.0)
            GL.glTexCoord2f(1.0, 1.0)
            GL.glVertex2f(1.0, 0.0)
            GL.glTexCoord2f(1.0, 0.0)
            GL.glVertex2f(1.0, -1.0)
            GL.glEnd()

    def _stereoRender(self):
        self.leftFBO.bind_texture(GL.GL_TEXTURE0)
        self._renderLeftFBO()
        self.leftFBO.unbind_texture()

        self.rightFBO.bind_texture(GL.GL_TEXTURE0)
        self._renderRightFBO()
        self.rightFBO.unbind_texture()

class SpannedWindow(MultiRenderWindow):

    """Create a display where left and right eye images are packed horizontally,
    preserving the aspect ratio.

    This is intended to provide multi-monitor support for extended desktop modes
    like Surround, TwinView, and Xinerama. This is the perfered method for 
    multi-display stereo. However, it is only supported on Windows and Linux 
    with the appropriate driver configuration.

    Use relfected=True if the image is reflected off a mirror, such as in a 
    mirror stereoscope.
    """

    def __init__(self, *args, **kwargs):
        self.reflected = kwargs.pop('reflected', False)
        MultiRenderWindow.__init__(self, *args, **kwargs)

    def _setupStereoFBO(self):
        # init framebuffer objects as render targets.
        # for this display mode, the framebuffers have half the horizontal 
        # resolution of the entire display.

        self.leftFBO = Framebuffer(self, (self.size[0]/2, self.size[1]))
        self.rightFBO = Framebuffer(self, (self.size[0]/2, self.size[1]))

        return True
    
    def _renderLeftFBO(self):
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(-1.0, -1.0)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex2f(0.0, 1.0)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex2f(0.0, -1.0)
        GL.glEnd()
    
    def _renderRightFBO(self):
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(0.0, -1.0)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex2f(0.0, 1.0)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glEnd()

    def _stereoRender(self):
        # blit left texture to the left side of screen
        self.leftFBO.bind_texture(GL.GL_TEXTURE0)
        # apply reflection if using a mirror stereoscope
        if self.reflected:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GL.glPushMatrix()
            GL.glScalef(-1, 1, 1)
            self._renderRightFBO() # use the right FBO quad
            GL.glPopMatrix()
            GL.glLoadIdentity()
        else:
            self._renderLeftFBO()

        self.leftFBO.unbind_texture() # not needed here?
        # blit right texture to the right side of screen
        self.rightFBO.bind_texture(GL.GL_TEXTURE0)
        if self.reflected:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GL.glPushMatrix()
            GL.glScalef(-1, 1, 1)
            self._renderLeftFBO() # use the left FBO quad
            GL.glPopMatrix()
            GL.glLoadIdentity()
        else:
            self._renderRightFBO()
        
        self.rightFBO.unbind_texture()

class AnaglyphWindow(MultiRenderWindow):

    """Create an anaglyph display for stereo viewing with colour glasses. Works
    on nearly all colour displays and provides a simple means of testing and
    demonstrating experiments without specialty hardware.

    NOTE: Only glasses with Red-Cyan lens are supported. There is no way (yet)
    to adjust the colours.
    """

    def __init__(self, *args, **kwargs):
        MultiRenderWindow.__init__(self, *args, **kwargs)

    def _prepareFBOrender(self):
        """Bind and configure the stereo shader. This is overridden by shaders
        needing more options."""
        GL.glUseProgram(self._progStereoShader)
        GL.glUniform1i(GL.glGetUniformLocation(self._progStereoShader,
            "leftEyeTexture"), 1)
        GL.glUniform1i(GL.glGetUniformLocation(self._progStereoShader,
            "rightEyeTexture"), 2)

    def _stereoRender(self):
        self.leftFBO.bind_texture(GL.GL_TEXTURE1)
        self.rightFBO.bind_texture(GL.GL_TEXTURE2)
        self._renderFBO()

    def _compileStereoShader(self):
        fragShader = '''
            uniform sampler2D leftEyeTexture;
            uniform sampler2D rightEyeTexture;

            void main() {
                vec4 leftEyeFrag = texture2D(leftEyeTexture,
                                             gl_TexCoord[0].st);
                vec4 rightEyeFrag = texture2D(rightEyeTexture,
                                              gl_TexCoord[0].st);

                // red-cyan is most commonly used (red=left, cyan=right)
                leftEyeFrag.rgba = vec4(1.0,
                                        leftEyeFrag.g,
                                        leftEyeFrag.b,
                                        leftEyeFrag.a);
                rightEyeFrag.rgba = vec4(rightEyeFrag.r,
                                         1.0,
                                         1.0,
                                         rightEyeFrag.a);

                gl_FragColor.rgba = vec4(leftEyeFrag.rgba * rightEyeFrag.rgba);
            }
            '''

        self._progStereoShader = _shaders.compileProgram(
            _shaders.vertSimple, fragShader)
