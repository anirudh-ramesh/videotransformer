#!/usr/bin/env python2

import cv2
import sys
import argparse
import random
import numpy as np


def nop(*args):
    pass

def main():
    parser = argparse.ArgumentParser(description='Process video files')
    parser.add_argument('vid_in', help='Input video file')
    parser.add_argument('--vid_out',  default='', help='Output video file')
    parser.add_argument('--resize', default='0x0', help='WxH')
    parser.add_argument('--grey', action='store_true', help='Convert to greyscale')
    parser.add_argument('--mirrorh', action='store_true', help='Mirror image horizontally')
    parser.add_argument('--mirrorv', action='store_true', help='Mirror image vertically')
    parser.add_argument('--noise-gaussian', default='0,0', metavar='M,D', help='Add gaussian noise (mean, stdev)')
    parser.add_argument('--noise-saltpepper', default='0,0', metavar='LT,HT', help='Add gaussian noise (low thresh, high thresh)')
    parser.add_argument('--invert-channels', default='n,n,n', metavar='?,?,?', help='Invert channel RGB (y/n)')
    parser.add_argument('--speedup', type=int, default=1, choices=xrange(1,5), help='Speed-up playback (integer factor)')
    parser.add_argument('--rand-frame-drop', type=int, default=0, metavar='PCT', help='Drop frames randomly (uniform(0,100) < PCT)')
    args = parser.parse_args()
    print repr(args)

    #arguments that require modifications
    try:
        outputFrameDimensions = args.resize.split('x')
        outputFrameDimensions = [int(i) for i in outputFrameDimensions]
        
    except ValueError:
        outputFrameDimensions = [0,0]

        
    try:
        noise_gaussian = args.noise_gaussian.split(',')
        noise_gaussian = [int(i) for i in noise_gaussian]
    except ValueError:
        noise_gaussian = (0.,0.)
        
    try:
        noise_saltpepper = args.noise_saltpepper.split(',')
        noise_saltpepper = [int(i) for i in noise_saltpepper]
    except ValueError:
        noise_saltpepper = (0.,0.)
        
    try:
        channel_inversions  = args.invert_channels.split(',')
        channel_inversions = [i == 'y' for i in channel_inversions]
    except ValueError:
        channel_inversions = [False, False, False]
        
    dropped_frame_factor = args.rand_frame_drop
    if not 0 < args.rand_frame_drop <= 100:
        dropped_frame_factor = 0
        
    ###########################

    cap = cv2.VideoCapture(args.vid_in)
    if not cap.isOpened():
        sys.stderr.write("Couldn't open video file.\n")
        exit(1)

    FPS = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    NB_FRAMES = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    INPUT_FRAME_WIDTH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    INPUT_FRAME_HEIGHT = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    sys.stderr.write('In: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT))

    outputFrameDimensions[0] = outputFrameDimensions[0] if outputFrameDimensions[0] > 0 else INPUT_FRAME_WIDTH
    outputFrameDimensions[1] = outputFrameDimensions[1] if outputFrameDimensions[1] > 0 else INPUT_FRAME_HEIGHT
    sys.stderr.write('Out: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, outputFrameDimensions[0], outputFrameDimensions[1]))

    if not FPS*NB_FRAMES*INPUT_FRAME_WIDTH*INPUT_FRAME_HEIGHT:
        sys.stderr.write("Video doesn't appear to be supported.\n")
        exit(1)

    if args.vid_out:
        sink = cv2.VideoWriter(args.vid_out, cv2.cv.FOURCC('M', 'J', 'P', 'G'), FPS, (outputFrameDimensions[0],outputFrameDimensions[1]), isColor=not args.grey)
        if not sink.isOpened():
            sys.stderr.write("Couldn't open output video file.\n")
            exit(1)

    cv2.namedWindow('Original')
    cv2.namedWindow('Modified')

    cv2.createTrackbar('Progress', 'Original', 0, NB_FRAMES, nop)
    cv2.createTrackbar('Progress', 'Modified', 0, NB_FRAMES, nop)

    for i in xrange(NB_FRAMES):
        keptFrame = True

        cv2.setTrackbarPos('Progress', 'Original', i)
        cv2.setTrackbarPos('Progress', 'Modified', i)
        (ret,inputImage) = cap.read()
        if not ret:
            sys.stderr.write("Couldn't decode frame.\n")
            break

        cv2.imshow('Original', inputImage)


        #Image transformations
        outputImage = inputImage

        #
        if INPUT_FRAME_WIDTH != outputFrameDimensions[0] or INPUT_FRAME_HEIGHT != outputFrameDimensions[1]:
            outputImage = cv2.resize(outputImage, tuple(outputFrameDimensions))

        #
        if args.mirrorh and args.mirrorv:
            outputImage = cv2.flip(outputImage, -1)
        elif args.mirrorh:
            outputImage = cv2.flip(outputImage, 1)
        elif args.mirrorv:
            outputImage = cv2.flip(outputImage, 0)

        #
        if noise_gaussian[0] and noise_gaussian[1]:
            chans = cv2.split(outputImage)
            noiseImage = chans[0].copy()
            cv2.randn(noiseImage, noise_gaussian[0], noise_gaussian[1])
            chans[0] = cv2.add(chans[0], noiseImage)
            chans[1] = cv2.add(chans[1], noiseImage)
            chans[2] = cv2.add(chans[2], noiseImage)
            outputImage = cv2.merge(chans)
        
        #
        if noise_saltpepper[0] and noise_saltpepper[1]:
            chans = cv2.split(outputImage)
            noiseImage = chans[0].copy()
            cv2.randu(noiseImage, 0, 255)
            black = noiseImage < noise_saltpepper[0]
            black = black * 255
            black = black.astype(np.uint8)
            white = noiseImage > noise_saltpepper[1]
            white = white * 255
            white = black.astype(np.uint8)
            chans[0] = cv2.subtract(chans[0], black)
            chans[0] = cv2.add(chans[0], white)
            chans[1] = cv2.subtract(chans[1], black)
            chans[1] = cv2.add(chans[1], white)
            chans[2] = cv2.subtract(chans[2], black)
            chans[2] = cv2.add(chans[2], white)
            outputImage = cv2.merge(chans)
        
        #
        if any(channel_inversions):
            chans = cv2.split(outputImage)
            
            #data is stored BGR
            if channel_inversions[2]:
                chans[0] = 255 - chans[0]
            if channel_inversions[1]:
                chans[1] = 255 - chans[1]
            if channel_inversions[0]:
                chans[2] = 255 - chans[2]
            outputImage = cv2.merge(chans)

        #
        if args.grey:
            outputImage = cv2.cvtColor(outputImage, cv2.cv.CV_BGR2GRAY)

        #
        keptFrame = random.randint(0, 100) >= dropped_frame_factor

        #
        if args.speedup > 1 and i % args.speedup > 0:
            keptFrame = False


        ############################################

        if keptFrame:
            cv2.imshow('Modified', outputImage)

            if args.vid_out:
                sink.write(outputImage)


        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            sys.stderr.write('Exitting.\n')
            break
        elif k > 0:
            sys.stderr.write('Unhandled key: {}\n'.format(k))

    cap.release()
    sink.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
