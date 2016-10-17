#!/usr/bin/env python2

"""
Stereo display wrappers and utilities. Written by Matthew Cutone (2016) with
code modified from visual.Window by PsychoPy contributers.
"""

import ctypes
import sys
import pyglet
import numpy
pyglet.options['debug_gl'] = True
GL = pyglet.gl

# PsychoPy imports
from . import shaders as _shaders
from . import window
from .. import platform_specific
from .. import logging
from . import globalVars

reportNDroppedFrames = 5  # stop raising warning after this

class Framebuffer(object):
    """Abstracted off-screen rendering targets for stereo renderings. Requires
    OpenGL 2.1+ to work properly. 
    """

    def __init__(self, width=800, height=600, depth=16, multiSample=False):
        
        # basic settings
        self.width = width
        self.height = height
        self.depth = depth

        # create the framebuffer
        self._initFramebuffer()

        # MSAA settings
        self._multiSample = multiSample
        self._numSamples = numSamples

        if self._multiSample:
            # create MSAA framebuffer if needed
            self._initMSframebuffer()

    def _initMSframebuffer(self):
        self.frameBufferMSid = GL.GLuint()
        GL.glGenFramebuffersEXT(1, ctypes.byref(self.frameBufferMSid))
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferMSid)

        # multisample color buffer
        self.textureMSid = GL.GLuint()
        GL.glGenTexturesEXT(1, ctypes.byref(self.textureMSid))
        GL.glBindTextureEXT(GL.GL_TEXTURE_2D_MULTISAMPLE, self.textureMSid)

        if self.depth == 8:
            useDepth = GL.GL_RGBA8
        elif self.depth == 16:
            useDepth = GL.GL_RGBA16

        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self._numSamples, 
            useDepth, int(w), int(h), GL.GL_TRUE)

        # attatch textures 
        GL.glFramebufferTexture2DEXT(GL.GL_FRAMEBUFFER_EXT, GL.GL_COLOR_ATTACHMENT0_EXT, 
            GL.GL_TEXTURE_2D_MULTISAMPLE, self.textureMSid, 0)

        # clear buffers
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # render buffer
        self.renderMSid = GL.GLuint()
        GL.glGenRenderbuffersEXT(1, ctypes.byref(self.renderMSid))
        GL.glBindRenderbufferEXT(GL.GL_RENDERBUFFER_EXT,  self.renderMSid)

        # the render buffer with multi-sampling
        GL.glRenderbufferStorageMultisampleEXT(GL.GL_RENDERBUFFER_EXT, self._numSamples, 
            GL.GL_DEPTH24_STENCIL8_EXT, int(w), int(h))  

        # attach the render buffer to the FBO
        GL.glFramebufferRenderbufferEXT(GL.GL_FRAMEBUFFER_EXT,
                                        GL.GL_STENCIL_ATTACHMENT_EXT,
                                        GL.GL_RENDERBUFFER_EXT,
                                        self.renderMSid)

        # status check to see if the framebuffer is complete
        status =  GL.glCheckFramebufferStatusEXT(GL.GL_FRAMEBUFFER_EXT)
        if status !=  GL.GL_FRAMEBUFFER_COMPLETE_EXT:
            if status == GL.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                print("cannot create multisample framebuffer")
            
            # unbind on failure and exit
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)

        GL.glDisable(GL.GL_TEXTURE_2D)
        # clear the buffers (otherwise the texture memory can contain
        # junk from other app)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    
    def _initFramebuffer(self):
        """Setup frambuffer object for off-screen rendering without multi-sampling.
        Setup is much like that in the window.Window class. This is likely to be
        only used as a target for MSAA resolution."""

        # create FBO
        self.frameBufferId = GL.GLuint()
        GL.glGenFramebuffersEXT(1, ctypes.byref(self.frameBufferId))
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)

        # Create texture to render to
        self.textureId = GL.GLuint()
        GL.glGenTextures(1, ctypes.byref(self.textureId))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureId)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)

        # configure texture settings
        if self.depth == 8:
            useDepth = GL.GL_RGBA8
        elif self.depth == 16:
            useDepth = GL.GL_RGBA16

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, useDepth,
                        int(self.width), int(self.height), 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        # attach texture to the frame buffer
        GL.glFramebufferTexture2DEXT(GL.GL_FRAMEBUFFER_EXT,
                                     GL.GL_COLOR_ATTACHMENT0_EXT,
                                     GL.GL_TEXTURE_2D, 
                                     self.textureId, 0)
        
        # clear buffers
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0_EXT)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                            
        # add a stencil buffer
        self.renderBufferId = GL.GLuint()
        GL.glGenRenderbuffersEXT(1, ctypes.byref(
            self.renderBufferId))  # like glGenTextures
        GL.glBindRenderbufferEXT(GL.GL_RENDERBUFFER_EXT, self.renderBufferId)
        GL.glRenderbufferStorageEXT(GL.GL_RENDERBUFFER_EXT,
                                    GL.GL_DEPTH24_STENCIL8_EXT,
                                    int(self.width), int(self.height))
        GL.glFramebufferRenderbufferEXT(GL.GL_FRAMEBUFFER_EXT,
                                        GL.GL_STENCIL_ATTACHMENT_EXT,
                                        GL.GL_RENDERBUFFER_EXT,
                                        self.renderBufferId)

        status = GL.glCheckFramebufferStatusEXT(GL.GL_FRAMEBUFFER_EXT)
        if status != GL.GL_FRAMEBUFFER_COMPLETE_EXT:
            logging.error("Error in framebuffer activation")
            # UNBIND THE FRAME BUFFER OBJECT THAT WE HAD CREATED
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)

        GL.glDisable(GL.GL_TEXTURE_2D)
        # clear the buffers (otherwise the texture memory can contain
        # junk from other app)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    def resolveMSAA(self, targetFBO):
        """Blit multisample texture to simple FBO texture for use"""
        #GL.glViewport(0, 0, self.size[0], self.size[1])
        GL.glBindFramebufferEXT(GL.GL_DRAW_FRAMEBUFFER_EXT, self.frameBufferMSid)
        GL.glBindFramebufferEXT(GL.GL_READ_FRAMEBUFFER_EXT, self.frameBufferId)
        GL.glDrawBuffer(self.GL.GL_COLOR_ATTACHMENT0_EXT)    
        GL.glBlitFramebufferEXT(0, 0, self.width, self.height, 0, 0, 
            self.width, self.height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    
    def bindTexture(self, where=GL.GL_TEXTURE0):
        pass
    
    def bindFBO(self, finalize=False):
        """Convienence function to bind FBO for read and draw. Set finalize to
        True resolve MSAA. Do this before blitting to back buffer."""
        if finalize and self._multiSample:
            # resolve MSAA and bind the simple texture to buffer
            self.resolveMSAA()
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)
        elif self._multiSample:
            # bind to the multisample buffer as the render target,
            # not being copied to render buffer yet.
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferMSid)
        else:
            # only simple texture being used
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.frameBufferId)
    
    def unbindFBO(self, toTarget=0):
        """Unbind to default framebuffer to default (0)"""
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, toTarget)
    
    def clearBuffer(self, buffer=GL.GL_COLOR_BUFFER_BIT):
        GL.glClear(buffer)

class MultiRenderWindow(window.Window):

    """
    Main support class for stereo modes that require off screen rendering for
    each eye's image; intended to provide alternative stereo display modes and
    stereo support for systems that do not support the GL_STEREO extension.

    This class should only be instanced by stereo wrappers or when implimenting
    a custom stereo window class. Using quad-buffers for stereo output is not
    permitted through this class, instead use the stereo=True option in the
    default visual.Window class.

    Usage:

        You can set which eye is the render target by calling the setBuffer()
        method and specifying either 'left', 'right', or 'both'. Subsiquent draw
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
         programmable fragment shaders is required (OpenGL 3.2+).
         Most modern graphics cards will have support for these functions.
         Ensure your graphics drivers are updated for best results.

    2.   Stimulus functions such as setAutoDraw() and mouse pointer hit checks
         behave unexpectedly when using stereo modes. It is best to call the
         draw() stimulus class method directly when rendering and writing your
         own click detection system.

    3.   Requires GPU support for anti-aliasing (OpenGL 3.2+).
    """

    def __init__(self, *args, **kwargs):

        # check for troublesome window settings
        if 'stereo' in kwargs:
            logging.warning("Requesting a quad-buffered stereo window is not "
                "allowed when using a 'MultiRenderWindow' sub-class. Disabling."
                )

            kwargs['stereo'] = False
        
        if 'aa_samples' in kwargs:
            if kwargs['aa_samples'] == None:
                max_samples = self.getMaxSamples()
                logging.warning("Multisample level not specified, setting to GL_MAX_SAMPLES"
                    " = {}".format(max_samples)
                    )
                self.aa_samples = max_samples
            else:
                self.aa_samples = kwargs['aa_samples']

            del kwargs['aa_samples']

        if 'useFBO' in kwargs:
            logging.warning("Using the built-in FBO support is not allowed "
            "when using MultiRenderWindow, disabling."
                )

            kwargs['useFBO'] = False
        
        self._size = kwargs['size']
        window.Window.__init__(self, *args, **kwargs)

    def getReservedColorAttachments(self):
        return([self.colorAttachmentLeft, self.colorAttachmentRight])

    def _endOfFlip(self, clearBuffer):
        """
        Clear both left and right eye framebuffer colour attachments if needed.
        """
        if clearBuffer:
            # clear the framebuffer color buffers
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.rightFBO)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.leftFBO)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

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

    def _resolveMSAA(self):
        """Blit multisample texture to simple FBO texture for use"""
        pass
        
    def _setupSimpleFBO(self, colAttach=GL.GL_COLOR_ATTACHMENT0_EXT, w=800, h=600):
        """Setup frambuffer object fir off-screen rendering without multi-sampling.
        Setup is much like that in the window.Window class. This is likely to be
        only used as a target for MSAA resolution."""

        # create FBO
        idxFBO = GL.GLuint()
        GL.glGenFramebuffersEXT(1, ctypes.byref(idxFBO))
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, idxFBO)

        # Create texture to render to
        idxTexture = GL.GLuint()
        GL.glGenTextures(1, ctypes.byref(idxTexture))
        GL.glBindTexture(GL.GL_TEXTURE_2D, idxTexture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F_ARB,
                        int(w), int(h), 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)
        # attach texture to the frame buffer
        GL.glFramebufferTexture2DEXT(GL.GL_FRAMEBUFFER_EXT,
                                     colAttach,
                                     GL.GL_TEXTURE_2D, 
                                     idxTexture, 0)
        
        # clear buffers
        GL.glReadBuffer(colAttach)
        GL.glDrawBuffer(colAttach)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                            
        # add a stencil buffer
        idxRender = GL.GLuint()
        GL.glGenRenderbuffersEXT(1, ctypes.byref(
            idxRender))  # like glGenTextures
        GL.glBindRenderbufferEXT(GL.GL_RENDERBUFFER_EXT, idxRender)
        GL.glRenderbufferStorageEXT(GL.GL_RENDERBUFFER_EXT,
                                    GL.GL_DEPTH24_STENCIL8_EXT,
                                     int(w), int(h))
        GL.glFramebufferRenderbufferEXT(GL.GL_FRAMEBUFFER_EXT,
                                        GL.GL_STENCIL_ATTACHMENT_EXT,
                                        GL.GL_RENDERBUFFER_EXT,
                                        idxRender)

        status = GL.glCheckFramebufferStatusEXT(GL.GL_FRAMEBUFFER_EXT)
        if status != GL.GL_FRAMEBUFFER_COMPLETE_EXT:
            logging.error("Error in framebuffer activation")
            # UNBIND THE FRAME BUFFER OBJECT THAT WE HAD CREATED
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)

        GL.glDisable(GL.GL_TEXTURE_2D)
        # clear the buffers (otherwise the texture memory can contain
        # junk from other app)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        return idxFBO, idxTexture, idxRender

    def _setupMultiSampleFBO(self, colAttach=GL.GL_COLOR_ATTACHMENT0_EXT, 
                             w=800, h=600):
        """Setup a multisample framebuffer as a render target. This will provide
        hardware accellerated anti-aliasing when drawing stimuli off-screen."""

        # create FBO
        idxFBO = GL.GLuint()
        GL.glGenFramebuffersEXT(1, ctypes.byref(idxFBO))
        GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, idxFBO)

        # multisample color buffer
        idxTexture = GL.GLuint()
        GL.glGenTexturesEXT(1, ctypes.byref(idxTexture))
        GL.glBindTextureEXT(GL.GL_TEXTURE_2D_MULTISAMPLE, idxTexture)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.aa_samples, 
            GL.GL_RGBA32F_ARB, int(w), int(h), GL.GL_TRUE)

        # render buffer
        idxRender = GL.GLuint()
        GL.glGenRenderbuffersEXT(1, ctypes.byref(idxRender))
        GL.glBindRenderbufferEXT(GL.GL_RENDERBUFFER_EXT, idxRender)
        # the render buffer with multi-sampling
        GL.glRenderbufferStorageMultisampleEXT(GL.GL_RENDERBUFFER_EXT, self.aa_samples, 
            GL.GL_DEPTH24_STENCIL8_EXT, int(w), int(h))  

        # attatch textures 
        GL.glFramebufferTexture2DEXT(GL.GL_FRAMEBUFFER_EXT, colAttach, 
            GL.GL_TEXTURE_2D_MULTISAMPLE, idxTexture, 0)
        
        # clear buffers
        GL.glReadBuffer(colAttach)
        GL.glDrawBuffer(colAttach)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # attach the render buffer to the FBO
        GL.glFramebufferRenderbufferEXT(GL.GL_FRAMEBUFFER_EXT,
                                        GL.GL_STENCIL_ATTACHMENT_EXT,
                                        GL.GL_RENDERBUFFER_EXT,
                                        idxRender)

        # status check to see if the framebuffer is complete
        status =  GL.glCheckFramebufferStatusEXT(GL.GL_FRAMEBUFFER_EXT)
        if status !=  GL.GL_FRAMEBUFFER_COMPLETE_EXT:
            if status == GL.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                print("cannot create multisample framebuffer")
            
            # unbind on failure and exit
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)

        GL.glDisable(GL.GL_TEXTURE_2D)
        # clear the buffers (otherwise the texture memory can contain
        # junk from other app)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        
        return idxFBO, idxTexture, idxRender

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

        # get base color attachment offset
        color_attach_base =  GL.GL_COLOR_ATTACHMENT0

        # allocate 3 attachments for stereo rendering near the end of the
        # range to reduce chances of interfering with user FBO color attachments
        maxAttach = ctypes.c_int(0)
        GL.glGetIntegerv(GL.GL_MAX_COLOR_ATTACHMENTS, ctypes.byref(maxAttach))

        # check if our driver can access at least 4 colour attachments
        if maxAttach.value < 4:
            print("Error not enough color attachments available.")
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER, 0)

            return False

        # allocate colour attachments as render targets
        self.leftDrawBuffer = color_attach_base + (maxAttach.value - 2)
        self.rightDrawBuffer = color_attach_base + (maxAttach.value - 1)
        self.leftMSAADrawBuffer = color_attach_base + (maxAttach.value - 4)
        self.rightMSAADrawBuffer = color_attach_base + (maxAttach.value - 3)
        #self.screenDrawBuffer = color_attach_base + (maxAttach.value - 1)

        # create C_UINT typed array for multi-buffer draw
        #colAttachList = [self.leftDrawBuffer, self.rightDrawBuffer]
        #self.bothDrawBuffers = ctypes.cast(
         #   (ctypes.c_uint * len(colAttachList))(*colAttachList),
          #  ctypes.POINTER(ctypes.c_uint))

        # get texture size of render target, some display modes need the render
        # target to be half the screen size (i.e. spanned window) since aspect
        # ratio needs to be maintained
        sizeFBO = self._getFBOSize()

        # create framebuffers for each eye
        self.leftMSAAFBO, self.leftMSAATexture, self.leftMSAARender = self._setupMultiSampleFBO(
            self.leftMSAADrawBuffer, sizeFBO[0], sizeFBO[1])
        self.rightMSAAFBO, self.rightMSAATexture, self.rightMSAARender = self._setupMultiSampleFBO(
            self.rightMSAADrawBuffer, sizeFBO[0], sizeFBO[1])

        self.leftFBO, self.leftTexture, self.leftRender = self._setupSimpleFBO(
            self.leftDrawBuffer, sizeFBO[0], sizeFBO[1])
        self.rightFBO, self.rightTexture, self.rightRender = self._setupSimpleFBO(
            self.rightDrawBuffer, sizeFBO[0], sizeFBO[1])

        #self.screenFBO, self.screenTexture, self.screenRender = setupFBO(self,
        #    self.screenDrawBuffer, self._size[0], self._size[1])

        return True

    def flip(self, clearBuffer=True):
        """
        * Copy of window.Window.flip() method with some modifications to better
        utilize the stereo rendering pipeline without the risk of breaking
        existing experiments. useFBO is prohibitted by the stereo extension
        and is not tested for or applied here. *

        Flip the front and back buffers after drawing everything for your
        frame. (This replaces the win.update() method, better reflecting what
        is happening underneath).

        If clearBuffer=True, the left and right eye framebuffers are cleared.
        The result of the stereo shaders completely overwrites the pixels in the
        back buffer so it does not need to be cleared.
        """

        for thisStim in self._toDraw:
            thisStim.draw()

        flipThisFrame = self._startOfFlip()
        if flipThisFrame:
            # set the viewport to span the whole screen are
            #L.glEnable(GL.GL_SAMPLE_ALPHA_TO_ONE)
            #self._setSize(fbo=False)

            # render the screen texture to the back buffer
            self._prepareFBOrender()
            #if self.bits != None:
            #    self.bits._prepareFBOrender()
            GL.glDisable(GL.GL_BLEND)
            stencilOn = GL.glIsEnabled(GL.GL_STENCIL_TEST)
            GL.glDisable(GL.GL_STENCIL_TEST)
            self._resolveMSAA()
            # unbind the framebuffer as the render target
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0)

            
            #GL.glDrawBuffer(GL.GL_BACK)
            #if self.bits != None:
            #    self.bits._prepareFBOrender()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glColor3f(1.0, 1.0, 1.0)  # glColor multiplies with texture
            GL.glColorMask(True, True, True, True)
            self._stereoRender() # call stereo rendering function   
            GL.glDisable(GL.GL_TEXTURE_2D)
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

    def setBuffer(self, buffer='left', clear=True):
        """
        Similar to setBuffer in Window, but changes the active eye FBO
        instead of the back quad-buffer. Buffers can be cleared individually
        if needed.
        """
        #GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if buffer == 'left':
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.leftMSAAFBO)
            self._setSize(fbo=True)

        elif buffer == 'right':
            GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, self.rightMSAAFBO)
            self._setSize(fbo=True)

        if clear:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    
    def _finishFBOrender(self):
        #GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glUseProgram(0)

    def _stereoRender(self):
        """Override with code to configure and execute transfering stereo
        buffers to textures and blitting them to the back buffer."""
        pass
    
    def _setSize(self, fbo):
        """Ensure the size of the FBO is returned when something gets
        Window.size. A bit of a hack, but will work well for now."""

        if fbo:
            self.size = self._getFBOSize()
        else:
            self.size = self._size

        GL.glViewport(0, 0, self.size[0], self.size[1])
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1, 1, -1, 1, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

"""
Below are the stereo window classes that produce various stereo displays. Each
one requires FBO support to function therefore are limited to OpenGL 2.1+. For
quad-buffer enabled through the GL_STEREO extension, use the stereo=True
option included in the vanilla Window class.
"""

class SpannedWindow(MultiRenderWindow):

    """Create a display where left and right eye images are packed horizontally
    preserving the aspect ratio of the image.

    This is intended to provide multi-monitor support for extended desktop modes
    like Surround, TwinView, and Xinerama. This is the perfered method for 
    multi-display stereo. However, this is only supported on Windows and Linux 
    with the supported drivers and configuration. 
    """

    def __init__(self, *args, **kwargs):
        # custom flags
        MultiRenderWindow.__init__(self, *args, **kwargs)

    def _compileStereoShader(self):
        # use the regular FBO rendering shader
        self._progStereoShader = self._progFBOtoFrame

    def _getFBOSize(self):
        return numpy.array((int(self._size[0]/2), int(self._size[1])), numpy.int)
    
    def _prepareFBOrender(self):
        GL.glUseProgram(self._progStereoShader)

    def _resolveMSAA(self):
        """Blit multisample texture to simple FBO texture for use"""
        #GL.glViewport(0, 0, self.size[0], self.size[1])
        self._setSize(fbo=True)
        GL.glBindFramebufferEXT(GL.GL_DRAW_FRAMEBUFFER_EXT, self.leftFBO)
        GL.glBindFramebufferEXT(GL.GL_READ_FRAMEBUFFER_EXT, self.leftMSAAFBO)
        GL.glDrawBuffer(self.leftDrawBuffer)    
        GL.glBlitFramebufferEXT(0, 0, self.size[0], self.size[1], 0, 0, self.size[0], self.size[1], 
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebufferEXT(GL.GL_DRAW_FRAMEBUFFER_EXT, self.rightFBO)
        GL.glBindFramebufferEXT(GL.GL_READ_FRAMEBUFFER_EXT, self.rightMSAAFBO)
        GL.glDrawBuffer(self.rightDrawBuffer)    
        GL.glBlitFramebufferEXT(0, 0, self.size[0], self.size[1], 0, 0, self.size[0], self.size[1], 
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        self._setSize(fbo=False)
    
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
        # blit left texture on the left side of screen
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.leftTexture)
        GL.glColorMask(True, True, True, True)
        self._renderLeftFBO()

        # blit right texture on the right side of screen
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rightTexture)
        GL.glColorMask(True, True, True, True)
        self._renderRightFBO()

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
        # set the viewport to span the whole screen area
        self._setSize(fbo=True)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.leftTexture)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rightTexture)

        self._renderFBO()

    def _resolveMSAA(self):
        """Blit multisample texture to simple FBO texture for use"""
        #GL.glViewport(0, 0, self.size[0], self.size[1])
        self._setSize(fbo=True)
        GL.glBindFramebufferEXT(GL.GL_DRAW_FRAMEBUFFER_EXT, self.leftFBO)
        GL.glBindFramebufferEXT(GL.GL_READ_FRAMEBUFFER_EXT, self.leftMSAAFBO)
        GL.glDrawBuffer(self.leftDrawBuffer)    
        GL.glBlitFramebufferEXT(0, 0, self.size[0], self.size[1], 0, 0, self.size[0], self.size[1], 
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebufferEXT(GL.GL_DRAW_FRAMEBUFFER_EXT, self.rightFBO)
        GL.glBindFramebufferEXT(GL.GL_READ_FRAMEBUFFER_EXT, self.rightMSAAFBO)
        GL.glDrawBuffer(self.rightDrawBuffer)    
        GL.glBlitFramebufferEXT(0, 0, self.size[0], self.size[1], 0, 0, self.size[0], self.size[1], 
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

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
