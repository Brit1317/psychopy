import sys
import os
import time
import re

from psychopy import logging, exceptions
from psychopy.constants import (PLAYING, PAUSED, FINISHED, STOPPED,
                                NOT_STARTED)
from psychopy.exceptions import SoundFormatError
from ._base import _SoundBase

import sounddevice as sd
import soundfile as sf

import numpy as np

travisCI = bool(str(os.environ.get('TRAVIS')).lower() == 'true')
logging.console.setLevel(logging.INFO)

logging.info("Loaded SoundDevice with {}".format(sd.get_portaudio_version()[1]))


def init():
    pass  # for compatibility with other backends


def getStreamLabel(sampleRate, channels, blockSize):
    """Returns the string repr of the stream label
    """
    return "{}_{}_{}".format(sampleRate, channels, blockSize)


class _StreamsDict(dict):
    """Keeps track of what streams have been created. On OS X we can have
    multiple streams under portaudio but under windows we can only have one.

    use the instance `streams` rather than creating a new instance of this
    """
    def getSimilar(self, sampleRate, channels=-1, blockSize=-1):
        """Do we already have a compatible stream?

        Many sounds can allow channels and blocksize to change but samplerate
        is generally fixed. Any values set to -1 above will be flexible. Any
        values set to an alternative number will be fixed

        usage:

            label, stream = streams.getSimilar(sampleRate=44100,  # must match
                                               channels=-1,  # any
                                               blockSize=-1)  # wildcard
        """
        label = getStreamLabel(sampleRate, channels, blockSize)
        # replace -1 with any regex integer
        simil = re.compile(label.replace("-1", r"[-+]?(\d+)"))  # I hate REGEX!
        for thisFormat in self.keys():
            if simil.match(thisFormat):  # we found a close-enough match
                return thisFormat, self[thisFormat]
        # if we've been given values in each place then create stream
        if (sampleRate not in [None, -1, 0] and
                channels not in [None, -1] and
                blockSize not in [None, -1]):
            return self.getStream(sampleRate, channels, blockSize)

    def getStream(self, sampleRate, channels, blockSize):
        """Gets a stream of exact match or returns a new one
        (if possible for the current operating system)
        """
        label = getStreamLabel(sampleRate, channels, blockSize)
        # try to retrieve existing stream of that name
        if label in self.keys():
            pass
        # on some systems more than one stream isn't supported so check
        elif sys.platform == 'win32' and len(self):
            raise exceptions.SoundFormatError(
                "Tried to create audio stream {} but {} already exists "
                "and {} doesn't support multiple portaudio streams"
                .format(label, self.keys()[0], sys.platform)
                )
        else:
            # create new stream
            self[label] = _SoundStream(sampleRate, channels, blockSize)
        return label, self[label]


streams = _StreamsDict()


class _SoundStream(object):
    def __init__(self, sampleRate, channels, blockSize, duplex=False):
        # initialise thread
        self.streams = []
        self.list = []
        # sound stream info
        self.sampleRate = sampleRate
        self.channels = channels
        self.duplex = duplex
        self.blockSize = blockSize
        self.sounds = []  # list of dicts for sounds currently playing
        if not travisCI:  # travis-CI testing does not have a sound device
            self._sdStream = sd.OutputStream(samplerate=sampleRate,
                                             blocksize=self.blockSize,
                                             latency='high',
                                             channels=channels,
                                             callback=self.callback)
            self._sdStream.start()
            self.device = self._sdStream.device
            self.latency = self._sdStream.latency
            self.cpu_load = self._sdStream.cpu_load
        self.frameN = 1
        self.takeTimeStamp = False
        self.frameTimes = range(5)  # DEBUGGING: store the last 5 callbacks
        self.lastFrameTime = time.time()
        self._tSoundRequestPlay = None

    def callback(self, toSpk, blockSize, timepoint, status):
        """This is a callback for the SoundDevice lib

        fromMic is data from the mic that can be extracted
        toSpk is a numpy array to be populated with data
        blockSize is the number of frames to be included each block
        timepoint has values:
            .currentTime
            .inputBufferAdcTime
            .outputBufferDacTime
        """
        if self.takeTimeStamp:
            logging.info("Entered callback: {} ms after last frame end"
                         .format((time.time()-self.lastFrameTime)*1000))
            logging.info("Entered callback: {} ms after sound start"
                         .format((time.time()-self._tSoundRequestPlay)*1000))
        t0 = time.time()
        self.frameN += 1
        toSpk *= 0  # it starts with the contents of the buffer before
        for thisSound in self.sounds:
            dat = thisSound._nextBlock()  # fetch the next block of data
            if self.channels == 2:
                toSpk[:len(dat), :] += dat  # add to out stream
            else:
                toSpk[:len(dat), 0] += dat  # add to out stream
            # check if that was a short block (sound is finished)
            if len(dat) < len(toSpk[:, :]):
                self.sounds.remove(thisSound)
                thisSound._EOS()
            # check if that took a long time
            t1 = time.time()
            if (t1-t0) > 0.001:
                logging.info("buffer_callback took {:.3f}ms that frame"
                             .format((t1-t0)*1000))
        self.frameTimes.pop(0)
        self.frameTimes.append(time.time()-self.lastFrameTime)
        self.lastFrameTime = time.time()
        if self.takeTimeStamp:
            logging.info("Callback durations: {}".format(self.frameTimes))
            logging.info("blocksize = {}".format(blockSize))
            self.takeTimeStamp = False

    def add(self, sound):
        t0 = time.time()
        self.sounds.append(sound)
        logging.info("took {} ms to add".format((time.time()-t0)*1000))

    def remove(self, sound):
        if sound in self.sounds:
            self.sounds.remove(sound)

    def __del__(self):
        print('garbage_collected_soundDeviceStream')
        if not travisCI:
            self._sdStream.stop()
        del self._sdStream
        sys.stdout.flush()


class SoundDeviceSound(_SoundBase):
    """Play a variety of sounds using the new SoundDevice library
    """
    def __init__(self, value="C", secs=0.5, octave=4, stereo=-1,
                 volume=1.0, loops=0,
                 sampleRate=44100, blockSize=128,
                 preBuffer=-1,
                 hamming=True,
                 startTime=0, stopTime=-1,
                 name='', autoLog=True):
        """
        :param value: note name ("C","Bfl"), filename or frequency (Hz)
        :param secs: duration (for synthesised tones)
        :param octave: which octave to use for note names (4 is middle)
        :param stereo: -1 (auto), True or False
                        to force sounds to stereo or mono
        :param volume: float 0-1
        :param loops: number of loops to play (-1=forever, 0=single repeat)
        :param sampleRate: sample rate for synthesized tones
        :param blockSize: the size of the buffer on the sound card
                         (small for low latency, large for stability)
        :param preBuffer: integer to control streaming/buffering
                           - -1 means store all
                           - 0 (no buffer) means stream from disk
                           - potentially we could buffer a few secs(!?)
        :param hamming: boolean (True to smooth the onset/offset)
        :param startTime: for sound files this controls the start of snippet
        :param stopTime: for sound files this controls the end of snippet
        :param name: string for logging purposes
        :param autoLog: whether to automatically log every change
        """
        self.sound = value
        self.name = name
        self.secs = secs  # for any synthesised sounds (notesand freqs)
        self.octave = octave  # for note name sounds
        self.stereo = stereo
        self.loops = loops
        self._loopsFinished = 0
        self.volume = volume
        self.hamming = hamming  # TODO: add hamming option
        self.startTime = startTime  # for files
        self.stopTime = stopTime  # for files specify thesection to be played
        self.blockSize = blockSize  # can be per-sound unlike other backends
        self.preBuffer = preBuffer
        self.frameN = 0
        self._tSoundRequestPlay = 0
        self.sampleRate = sampleRate
        self.channels = None
        self.duplex = None
        self.autoLog = autoLog
        self.streamLabel = ""
        self.sourceType = 'unknown'  # set to be file, array or freq
        self.sndFile = None
        self.sndArr = None
        # setSound (determines sound type)
        self.setSound(value)
        self.status = NOT_STARTED

    def setSound(self, value, secs=0.5, octave=4, hamming=True, log=True):
        """Set the sound to be played.

        Often this is not needed by the user - it is called implicitly during
        initialisation.

        :parameters:

            value: can be a number, string or an array:
                * If it's a number between 37 and 32767 then a tone will
                  be generated at that frequency in Hz.
                * It could be a string for a note ('A', 'Bfl', 'B', 'C',
                  'Csh'. ...). Then you may want to specify which octave.
                * Or a string could represent a filename in the current
                  location, or mediaLocation, or a full path combo
                * Or by giving an Nx2 numpy array of floats (-1:1) you can
                  specify the sound yourself as a waveform

            secs: duration (only relevant if the value is a note name or
                a frequency value)

            octave: is only relevant if the value is a note name.
                Middle octave of a piano is 4. Most computers won't
                output sounds in the bottom octave (1) and the top
                octave (8) is generally painful
        """
        # start with the base class method
        _SoundBase.setSound(self, value, secs, octave, hamming, log)
        try:
            label, s = streams.getStream(sampleRate=self.sampleRate,
                                         channels=self.channels,
                                         blockSize=self.blockSize)
        except SoundFormatError as err:
            # try to use something similar (e.g. mono->stereo)
            # then check we have an appropriate stream open
            altern = streams.getSimilar(sampleRate=self.sampleRate,
                                        channels=-1,
                                        blockSize=-1)
            if altern is None:
                raise SoundFormatError(err)
            else:  # safe to extract data
                label, s = altern
            # update self in case it changed to fit the stream
            self.sampleRate = s.sampleRate
            self.channels = s.channels
            self.blockSize = s.blockSize
        self.streamLabel = label

    def _setSndFromFile(self, filename):
        self.sndFile = f = sf.SoundFile(filename)
        self.sourceType = 'file'
        self.sampleRate = f.samplerate
        self.channels = f.channels
        info = sf.info(filename)  # needed for duration?
        # process start time
        if self.startTime and self.startTime > 0:
            startFrame = self.startTime*self.sampleRate
            self.sndFile.seek(int(startFrame))
            self.t = self.startTime
        else:
            self.t = 0
        # process stop time
        if self.stopTime and self.stopTime > 0:
            requestedDur = self.stopTime - self.t
            maxDur = info.duration
            self.duration = min(requestedDur, maxDur)
        else:
            self.duration = info.duration - self.t
        # can now calculate duration in frames
        self.durationFrames = int(round(self.duration*self.sampleRate))
        # are we preloading or streaming?
        if self.preBuffer == 0:
            # no buffer - stream from disk on each call to nextBlock
            pass
        elif self.preBuffer == -1:
            # no buffer - stream from disk on each call to nextBlock
            sndArr = self.sndFile.read(frames=self.durationFrames)
            self.sndFile.close()
            self._setSndFromArray(sndArr)

    def _setSndFromFreq(self,  thisFreq, secs, hamming=True):
        self.freq = thisFreq
        self.secs = secs
        self.sourceType = 'freq'
        self.t = 0
        self.duration = self.secs
        if hamming:
            logging.warning(
                "Hamming smoothing not yet implemented for SoundDeviceSound."
                )

    def _setSndFromArray(self, thisArray):
        """For pysoundcard all sounds are ultimately played as an array so
        other setSound methods are going to call this having created an arr
        """

        if self.stereo and thisArray.ndim == 1:
            # make mono sound stereo
            self.sndArr = np.resize(thisArray, [2, len(thisArray)]).T
        else:
            self.sndArr = np.asarray(thisArray)
        self._nSamples = thisArray.shape[0]
        # set to run from the start:
        self.seek(0)

    def play(self, loops=None):
        """Start the sound playing
        """
        if loops is not None and self.loops != loops:
            self.setLoops(loops)
        self.status = PLAYING
        self._tSoundRequestPlay = time.time()
        streams[self.streamLabel].takeTimeStamp = True
        streams[self.streamLabel].add(self)

    def pause(self):
        """Stop the sound but play will continue from here if needed
        """
        self.status = PAUSED
        streams[self.streamLabel].remove(self)

    def stop(self):
        """Stop the sound and return to beginning
        """
        streams[self.streamLabel].remove(self)
        self.seek(0)
        self.status = STOPPED

    def _nextBlock(self):
        framesLeft = (self.stopTime-self.t)*self.sampleRate
        nFrames = min(self.blockSize, framesLeft)
        if self.sourceType == 'file' and self.preBuffer == 0:
            # streaming sound block-by-block direct from file
            block = self.sndFile.read(nFrames)
            # TODO: check if we already finished using sndFile?

        elif (self.sourceType == 'file' and self.preBuffer == -1) \
            or self.sourceType == 'array':
            # An array, or a file entirely loaded into an array
            ii = int(round(self.t * self.sampleRate))
            if self.stereo == 1:  # don't treat as boolean. Might be -1
                block = self.sndArr[ii:ii+nFrames, :]
            elif self.stereo == 0:
                block = self.sndArr[ii:ii+nFrames]

        elif self.sourceType == 'freq':
            startT = self.t
            stopT = self.t+self.blockSize/float(self.sampleRate)
            xx = np.linspace(
                start=startT*self.freq*2*np.pi,
                stop=stopT*self.freq*2*np.pi,
                num=self.blockSize, endpoint=False
                )
            block = np.sin(xx)
            # if run beyond our desired t then set to zeros
            if stopT > self.secs:
                tRange = np.linspace(startT, stopT,
                                     num=self.blockSize, endpoint=False)
                block[tRange > self.secs] == 0
                # and inform our EOS function that we finished
                self._EOS()

        self.t += self.blockSize/float(self.sampleRate)
        return block

    def seek(self, t):
        self.t = t
        self.frameN = int(round(t * self.sampleRate))
        if self.sndFile and not self.sndFile.closed:
            self.sndFile.seek(self.frameN)

    def _EOS(self):
        """Function called on End Of Stream
        """
        self._loopsFinished += 1
        if self.loops == 0:
            self.stop()
        elif self.loops > 0 and self._loopsFinished >= self.loops:
            self.stop()

        streams[self.streamLabel].remove(self)
        self.status = FINISHED

    @property
    def stream(self):
        """Read-only property returns the the stream on which the sound
        will be played
        """
        return streams[self.streamLabel]
