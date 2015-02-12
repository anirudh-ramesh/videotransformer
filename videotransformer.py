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
    parser.add_argument('--noise-saltpepper', default='0,0', metavar='SPCT,PPCT', help='Add salt-pepper noise (salt pct, pepper pct)')
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

    if args.reverse and args.seek:
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

    sink = None
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

    TEMPORAL_LEN = 25
    subPix = [[[[] for h in xrange(outputFrameDimensions[1])] for w in xrange(outputFrameDimensions[0])] for t in xrange(-TEMPORAL_LEN, TEMPORAL_LEN+1)]
    singularities = []

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
        if noise_saltpepper[0] or noise_saltpepper[1]:
            chans = cv2.split(outputImage)
            
            whiteNoiseImage = chans[0].copy()
            cv2.randu(whiteNoiseImage, 0, 100)
            white = whiteNoiseImage < noise_saltpepper[0]
            white = white.astype(np.uint8)
            white *= 255

            blackNoiseImage = chans[0].copy()
            cv2.randu(blackNoiseImage, 0, 100)
            black = blackNoiseImage < noise_saltpepper[1]
            black = black.astype(np.uint8)
            black *= 255
            black = cv2.bitwise_not(black) #we want all 0's

            chans[0] = cv2.bitwise_and(chans[0], black)
            chans[0] = cv2.add(chans[0], white)
            chans[1] = cv2.bitwise_and(chans[1], black)
            chans[1] = cv2.add(chans[1], white)
            chans[2] = cv2.bitwise_and(chans[2], black)
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
        destPix = None
        if args.grey:
            outputImage = cv2.cvtColor(outputImage, cv2.cv.CV_BGR2GRAY)

            if args.smudge:
                PX_GAIN = 500
                NORM_PX_SIGMA = 15
                NORM_PX_MU = 0

                T_GAIN = 20000
                NORM_T_SIGMA = 15
                NORM_T_MU = 0

                destPix = outputImage.copy() * 0

                #Flatten subPixels for the frame coming out of the queue
                for w in xrange(outputFrameDimensions[0]):
                    for h in xrange(outputFrameDimensions[1]):
                        a = subPix[0][w][h]

                        if len(a) > 0:
                            destPix[h][w] = np.uint8(reduce(lambda x, y: x + y, a) / len(a))

                #Fill in the blanks in the image we've created. Obtain value from neighboring x,y pixels
                for w in xrange(outputFrameDimensions[0]):
                    for h in xrange(outputFrameDimensions[1]):
                        total = 0
                        count = 0
                        if len(subPix[0][w][h]) == 0:
                            for w1 in xrange(w-3, w+4):
                                for h1 in xrange(h-3, h+4):
                                    if 0 <= w1 < outputFrameDimensions[0] and 0 <= h1 < outputFrameDimensions[1] and len(subPix[0][w1][h1]) != 0:
                                        count = count + 1
                                        total += destPix[h1][w1]
                            avg = (total / count) if count > 0 else 0
                            destPix[h][w] = np.uint8(avg)

                #we're done with the subPix frame exiting the queue
                del(subPix[0])

                #shift the subPix frames to the left
                subPix.append([[[] for h in xrange(outputFrameDimensions[1])] for w in xrange(outputFrameDimensions[0])])

                #we're not ready to handle the new frame

                #Remove old singularities that can't apply anymore
                singularities = [x for x in singularities if (i - x['start']) <= TEMPORAL_LEN]

                #Perhaps add a new singularity
                if random.randint(0, 100) < 5: #% probability of adding a new singularity to any given frame
                    #TODO: attach a varying gain to t and (x,y) to each singularity
                    #select a random point in (x,y) space (the origin of the singularity) and a unit vector in (t,x,y) space describing its effect
                    oPoint = np.array([random.randint(0,outputFrameDimensions[0]), random.randint(0,outputFrameDimensions[1])])
                    oVector = np.array([random.randint(-TEMPORAL_LEN, TEMPORAL_LEN), random.randint(0,outputFrameDimensions[0]), random.randint(0,outputFrameDimensions[1])])
                    #print('Singularity is {} with direction {}'.format(oPoint, oVector))
                    oVector = np.divide(oVector, np.linalg.norm(oVector)) #make it a unit vector
                    #print('Singularity is {} with direction {}'.format(oPoint, oVector))
                    singularities.append({'start': i, 'oPoint': oPoint, 'oVector': oVector})

                #Calculate pixel movements for all pixels in range of all active singularities
                #Each singularities affects each pixel in the image independently
                if len(singularities) == 0:
                    #put everything in the middle of the buffer; no anomalities to push the pixels around
                    ttgt = TEMPORAL_LEN
                    xtgt = w
                    ytgt = h
                    for w in xrange(outputFrameDimensions[0]):
                        for h in xrange(outputFrameDimensions[1]):
                            subPix[ttgt][xtgt][ytgt].append(np.uint32(outputImage[h][w])) #inverted image storage
                else:
                    for singularity in singularities:
                        singStart = singularity['start']
                        singOPoint = singularity['oPoint']
                        singOVector = singularity['oVector'] #t,x,y

                        cv2.circle(destPix, tuple(singOPoint), 6, (255,255,255)) #show singularities
                        cv2.waitKey(1)

                        for w in xrange(outputFrameDimensions[0]):
                            for h in xrange(outputFrameDimensions[1]):
                                dist = np.linalg.norm([singStart-i, singOPoint[0]-w, singOPoint[1]-h])

                                pxMovementMag = PX_GAIN * normalPDF(NORM_PX_SIGMA, NORM_PX_MU, dist)
                                tMovementMag = T_GAIN * normalPDF(NORM_T_SIGMA, NORM_T_MU, dist)

                                movement = np.array([int(tMovementMag*singOVector[0]), int(pxMovementMag*singOVector[1]), int(pxMovementMag*singOVector[2])]) #t,x,y

                                xyDest = np.array([w,h]) + movement[1:]
                                #print('Pixel {} is moved by {} to {}'.format([w,h], movement, xyDest))

                                ttgt = TEMPORAL_LEN+movement[0]
                                xtgt = xyDest[0]
                                ytgt = xyDest[1]
                                if 0 <= ttgt <= 2*TEMPORAL_LEN and 0 <= xtgt < outputFrameDimensions[0] and 0 <= ytgt < outputFrameDimensions[1]: 
                                    subPix[ttgt][xtgt][ytgt].append(np.uint32(outputImage[h][w])) #inverted image storage
                                else:
                                    #print('pixel out of the frame {} {} {}'.format(ttgt, xtgt, ytgt))
                                    pass

                outputImage = destPix

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

        sys.stderr.write('Progress: %.3f %%            \r' % (i*100./NB_FRAMES))


        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            sys.stderr.write('Exitting.\n')
            break
        elif k > 0:
            sys.stderr.write('Unhandled key: {}\n'.format(k))

    cap.release()
    if sink:
        sink.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
