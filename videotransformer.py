#!/usr/bin/env python2

import cv2
import sys
import argparse
import random
import numpy as np


def nop(*args):
    pass

def normalPDF(sigma, mu, x):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*(sigma**2)))

def main():
    parser = argparse.ArgumentParser(description='Process video files')
    parser.add_argument('video_input', help='Input video file')
    parser.add_argument('--video-output',  default='', help='Output video file')
    parser.add_argument('--seek',  type=int, default=0, help='Start at specific frame')
    parser.add_argument('--batch',  action='store_true', help='Do not show GUI')
    parser.add_argument('--resize', default='0x0', help='WxH')
    parser.add_argument('--reverse', action='store_true', help='Process video in reverse')
    parser.add_argument('--grey', action='store_true', help='Convert to greyscale')
    parser.add_argument('--mirrorh', action='store_true', help='Mirror image horizontally')
    parser.add_argument('--mirrorv', action='store_true', help='Mirror image vertically')
    parser.add_argument('--noise-gaussian', default='0,0', metavar='M,D', help='Add gaussian noise (mean, stdev)')
    parser.add_argument('--noise-saltpepper', default='0,0', metavar='LT,HT', help='Add gaussian noise (low thresh, high thresh)')
    parser.add_argument('--invert-channels', default='n,n,n', metavar='?,?,?', help='Invert channel RGB (y/n)')
    parser.add_argument('--speedup', type=int, default=1, choices=xrange(1,5), help='Speed-up playback (integer factor)')
    parser.add_argument('--rand-frame-drop', type=int, default=0, metavar='PCT', help='Drop frames randomly (uniform(0,100) < PCT)')
    parser.add_argument('--smudge', action='store_true', help='Activate smudge generation')
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
    
    if args.smudge and not args.grey:
        sys.stderr.write("ERROR: --smudge can only be used in combination with --grey\n")
        exit(1)
        
    if args.reverse and not args.seek:
        sys.stderr.write("ERROR: --reverse is incompatible with --seek at this time\n")
        exit(1)
    ###########################

    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        sys.stderr.write("Couldn't open video file.\n")
        exit(1)

    FPS = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    NB_FRAMES = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    INPUT_FRAME_WIDTH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    INPUT_FRAME_HEIGHT = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    sys.stderr.write('In: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT))
    
    if 0 <= args.seek < NB_FRAMES:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, args.seek)
    else:
        sys.stderr.write('Could not seek past end of video\n')

    outputFrameDimensions[0] = outputFrameDimensions[0] if outputFrameDimensions[0] > 0 else INPUT_FRAME_WIDTH
    outputFrameDimensions[1] = outputFrameDimensions[1] if outputFrameDimensions[1] > 0 else INPUT_FRAME_HEIGHT
    sys.stderr.write('Out: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, outputFrameDimensions[0], outputFrameDimensions[1]))

    if not FPS*NB_FRAMES*INPUT_FRAME_WIDTH*INPUT_FRAME_HEIGHT:
        sys.stderr.write("Video doesn't appear to be supported.\n")
        exit(1)

    if args.video_output:
        sink = cv2.VideoWriter(args.video_output, cv2.cv.FOURCC('M', 'J', 'P', 'G'), FPS, (outputFrameDimensions[0],outputFrameDimensions[1]), isColor=not args.grey)
        if not sink.isOpened():
            sys.stderr.write("Couldn't open output video file.\n")
            exit(1)

    if not args.batch:
        cv2.namedWindow('Original')
        cv2.namedWindow('Modified')
        cv2.createTrackbar('Progress', 'Original', 0, NB_FRAMES, nop)
        cv2.createTrackbar('Progress', 'Modified', 0, NB_FRAMES, nop)

    for i in xrange(NB_FRAMES):
        keptFrame = True

        if not args.batch:
            cv2.setTrackbarPos('Progress', 'Original', i)
            cv2.setTrackbarPos('Progress', 'Modified', i)

        if args.reverse:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, NB_FRAMES-i-1)
        
        (ret,inputImage) = cap.read()
        if not ret:
            sys.stderr.write("Couldn't decode frame.\n")
            break

        if not args.batch:
            cv2.imshow('Original', inputImage)


        #Image transformations
        outputImage = inputImage.copy()

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
            
            if args.smudge:
                oPoint = np.array([290, 50])
                oVector = np.array([-0.707,-0.707]) #should be a unit vector
                maxDist = 75 #px
                gain = 500
                normSigma = 15
                normMu = 0
                
                #create subpixel array
                subPix = [[[] for h in xrange(outputFrameDimensions[1])] for w in xrange(outputFrameDimensions[0])]
                 
                destPix = outputImage.copy()
                destPix *= 0
 
                for w in xrange(outputFrameDimensions[0]):
                    cv2.waitKey(1)
                    for h in xrange(outputFrameDimensions[1]):
                        dist = np.linalg.norm(oPoint - np.array([w,h]))
                        if dist > maxDist:
                            subPix[w][h].append(outputImage[h][w]) #inverted image storage
                            continue
                         
                        pxMovementMag = gain * normalPDF(normSigma, normMu, dist)
                        pxMovement = oVector * np.array([pxMovementMag, pxMovementMag])
                        dest = np.array([w,h]) + pxMovement
                        
 
                        if 0 <= dest[0] < outputFrameDimensions[0] and 0 <= dest[1] < outputFrameDimensions[1]: 
                            subPix[int(dest[0])][int(dest[1])].append(np.uint32(outputImage[h][w])) #inverted image storage
                     
                #flatten output image
                for w in xrange(outputFrameDimensions[0]):
                    for h in xrange(outputFrameDimensions[1]):
                        a = subPix[w][h]
                        
                        if len(a) > 0:
                            destPix[h][w] = np.uint8(reduce(lambda x, y: x + y, a) / len(a))
                                    
                #fill in the blanks in the image
                #obtain value from neighboring pixels
                
                for w in xrange(outputFrameDimensions[0]):
                    for h in xrange(outputFrameDimensions[1]):
                        total = 0
                        count = 0
                        if len(subPix[w][h]) == 0:
                            for w1 in xrange(w-3, w+4):
                                for h1 in xrange(h-3, h+4):
                                    if 0 <= w1 < outputFrameDimensions[0] and 0 <= h1 < outputFrameDimensions[1] and len(subPix[w1][h1]) != 0:
                                        count = count + 1
                                        total += destPix[h1][w1]
                            avg = (total / count) if count > 0 else 0
                            destPix[h][w] = np.uint8(avg)
                                
                                    
                                
                                    
                                    
 
                outputImage = destPix
                #cv2.circle(outputImage, tuple(oPoint), 5, 255)

        #
        keptFrame = random.randint(0, 100) >= dropped_frame_factor

        #
        if args.speedup > 1 and i % args.speedup > 0:
            keptFrame = False


        ############################################

        if keptFrame:
            modifiedImage = outputImage.copy()
            if not args.batch:
                cv2.imshow('Modified', outputImage)

            if args.video_output:
                sink.write(modifiedImage)


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
