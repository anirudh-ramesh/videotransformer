#!/usr/bin/env python2

import cv2
import sys
import argparse
import random
import numpy as np
import math

def nop(*args):
    pass

def normalPDF(sigma, mu, x):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*(sigma**2)))

def main():
    parser = argparse.ArgumentParser(description='Process video files')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--output',  default='', help='Output video file')
    parser.add_argument('--seek',  type=int, default=0, help='Start at specific frame')
    parser.add_argument('--batch',  action='store_true', help='Do not show GUI')
    parser.add_argument('--resize', default='0x0', help='WxH')
    parser.add_argument('--reverse', action='store_true', help='Process video in reverse')
    parser.add_argument('--grey', action='store_true', help='Convert to greyscale')
    parser.add_argument('--mirrorh', action='store_true', help='Mirror image horizontally')
    parser.add_argument('--mirrorv', action='store_true', help='Mirror image vertically')
    parser.add_argument('--noise-gaussian', default='0', metavar='sigma', help='Add gaussian noise (sigma)')
    parser.add_argument('--noise-saltpepper', default='0,0', metavar='SPCT,PPCT', help='Add salt-pepper noise (salt pct, pepper pct)')
    parser.add_argument('--invert-channels', default='n,n,n', metavar='?,?,?', help='Invert channel RGB (y/n)')
    parser.add_argument('--speedup', type=int, default=1, choices=xrange(1,5), help='Speed-up playback (integer factor)')
    parser.add_argument('--rand-frame-drop', type=int, default=0, metavar='PCT', help='Drop frames randomly (uniform(0,100) < PCT)')
    parser.add_argument('--smudge', default='0,0,0,0', metavar='pctNewSing,sigN,sigT,sigPxDispl', help='Activate smudge generation (pctNewSing,sigmaNeigh,sigmaTime,sigmaPxDispl)')
    # Add shear
    parser.add_argument('--shear', type=int, default=0, choices=xrange(0,20), metavar='Angle', help='Shear image (angle)')
    # Add stretch
    parser.add_argument('--stretch', default='0x0', help='WxH')
    #
    args = parser.parse_args()
    print repr(args)

    #arguments that require modifications
    try:
        outputFrameDimensions = args.resize.split('x')
        outputFrameDimensions = [int(i) for i in outputFrameDimensions]
    except ValueError:
        outputFrameDimensions = [0,0]

    # Add stretch
    try:
        outputFrameDimensions = args.stretch.split('x')
        outputFrameDimensions = [int(i) for i in outputFrameDimensions]
    except ValueError:
        outputFrameDimensions = [0,0]
    #
    
    try:
        noise_gaussian_sigma = args.noise_gaussian.split(',')
        noise_gaussian_sigma = [int(i) for i in noise_gaussian_sigma]
    except ValueError:
        noise_gaussian_sigma = 0.

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
        
    try:
        smudge = args.smudge.split(',')
        smudge = [float(i) for i in smudge]
        smudge = {'pctNewSing': smudge[0], 'sigma_neigh': smudge[1], 'sigma_time': smudge[2], 'sigma_px_displ': smudge[3]}
    except ValueError:
        smudge = {'pctNewSing': 0, 'sigma_neigh': 0., 'sigma_time': 0., 'sigma_px_displ': 0.}

    if any(smudge.values()) and not args.grey:
        sys.stderr.write("ERROR: --smudge can only be used in combination with --grey\n")
        exit(1)

    if args.reverse and args.seek:
        sys.stderr.write("ERROR: --reverse is incompatible with --seek at this time\n")
        exit(1)
    ###########################

    cap = cv2.VideoCapture(args.input)
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

    # Add shear (insert code away from rest since input frame dimensions are required)
    try:
        if args.shear:
            shearAngle = args.shear
            shearFactor = math.sin(args.shear*np.pi/180)
            outputFrameDimensions[0] = int(INPUT_FRAME_WIDTH * (shearFactor + 1))
            outputFrameDimensions[1] = INPUT_FRAME_HEIGHT
    except ValueError:
        outputFrameDimensions = [0,0]
    #
    
    outputFrameDimensions[0] = outputFrameDimensions[0] if outputFrameDimensions[0] > 0 else INPUT_FRAME_WIDTH
    outputFrameDimensions[1] = outputFrameDimensions[1] if outputFrameDimensions[1] > 0 else INPUT_FRAME_HEIGHT
    sys.stderr.write('Out: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, outputFrameDimensions[0], outputFrameDimensions[1]))

    if not FPS*NB_FRAMES*INPUT_FRAME_WIDTH*INPUT_FRAME_HEIGHT:
        sys.stderr.write("Video doesn't appear to be supported.\n")
        exit(1)

    sink = None
    if args.output:
        sink = cv2.VideoWriter(args.output, cv2.cv.FOURCC('M', 'J', 'P', 'G'), FPS, (outputFrameDimensions[0],outputFrameDimensions[1]), isColor=not args.grey)
        if not sink.isOpened():
            sys.stderr.write("Couldn't open output video file.\n")
            exit(1)

    if not args.batch:
        cv2.namedWindow('Original')
        cv2.namedWindow('Modified')
        cv2.createTrackbar('Progress', 'Original', 0, NB_FRAMES, nop)
        cv2.createTrackbar('Progress', 'Modified', 0, NB_FRAMES, nop)

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
        # Modify resize
        if args.resize and (INPUT_FRAME_WIDTH != outputFrameDimensions[0] or INPUT_FRAME_HEIGHT != outputFrameDimensions[1]):
            outputImage = cv2.resize(outputImage, tuple(outputFrameDimensions))

        # Add shear
        if args.shear:
            rows, columns = outputImage.shape[:2]
            sourceTriangle = np.array([(0,0),(columns-1,0),(0,rows-1)], np.float32)
            destinationTriangle = np.array([(0,0),(columns-1,0), (columns*shearFactor,rows-1)],np.float32)
            warpMatrix = cv2.getAffineTransform(sourceTriangle,destinationTriangle)
            outputImage = cv2.warpAffine(outputImage,warpMatrix,(int(columns*shearFactor)+columns,rows))

        # Add stretch
        if args.stretch:
            rows, columns = outputImage.shape[:2]
            sourceTriangle = np.array([(0,0),(columns-1,0),(0,rows-1)], np.float32)
            destinationTriangle = np.array([(0,0),(outputFrameDimensions[0]-1,0), (0,outputFrameDimensions[1]-1)],np.float32)
            warpMatrix = cv2.getAffineTransform(sourceTriangle,destinationTriangle)
            outputImage = cv2.warpAffine(outputImage,warpMatrix,tuple(outputFrameDimensions))
        
        #
        if args.mirrorh and args.mirrorv:
            outputImage = cv2.flip(outputImage, -1)
        elif args.mirrorh:
            outputImage = cv2.flip(outputImage, 1)
        elif args.mirrorv:
            outputImage = cv2.flip(outputImage, 0)

        #
        if noise_gaussian_sigma[0] and noise_gaussian_sigma[1]:
            chans = cv2.split(outputImage)
            noiseImage = chans[0].copy()
            cv2.randn(noiseImage, noise_gaussian_sigma[0], noise_gaussian_sigma[1])
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
                MAX_SINGULARITY_DURATION = int(3 * smudge['sigma_time']) #99% of the effect is captured in this range

                subPixels = [[[] for h in xrange(outputFrameDimensions[1])] for w in xrange(outputFrameDimensions[0])]
                destPix = outputImage.copy() * 0

                #Remove old singularities that can't apply anymore
                singularities = [x for x in singularities if (i - x['start']) <= MAX_SINGULARITY_DURATION]

                #Perhaps add a new singularity
                if random.randint(0, 100) < smudge['pctNewSing']: #% probability of adding a new singularity to any given frame
                    #select a random point in (x,y) space (the origin of the singularity) and a unit vector in (t,x,y) space describing its effect
                    threePxDisplSigmas = 3*smudge['sigma_px_displ']
                    #we don't want a point too close to an edge
                    oPoint = np.array([random.randint(threePxDisplSigmas,outputFrameDimensions[0]-threePxDisplSigmas), \
                                       random.randint(threePxDisplSigmas,outputFrameDimensions[1]-threePxDisplSigmas)])

                    oVector = np.array([random.randint(0,outputFrameDimensions[0]), random.randint(0,outputFrameDimensions[1])])
                    oVector = np.divide(oVector, np.linalg.norm(oVector)) #make it a unit vector
                    oStart = i + MAX_SINGULARITY_DURATION # anomaly can't start right away otherwise it will create a sudden effect, push it in the future
                    singularities.append({'start': oStart, 'oPoint': oPoint, 'oVector': oVector})

                
                if len(singularities) == 0:
                    destPix = outputImage
                #Calculate pixel movements for all pixels in range of all active singularities
                else:
                    for w in xrange(outputFrameDimensions[0]):
                        for h in xrange(outputFrameDimensions[1]):
                            movement = [0.,0.] #x,y
                            for singularity in singularities:
                                dist = np.linalg.norm([singularity['oPoint'] - [w,h]])

                                distNormalAt0 = normalPDF(smudge['sigma_px_displ'], 0, 0)
                                distNormalAtDist = normalPDF(smudge['sigma_px_displ'], 0, dist)
                                timeNormalAt0 = normalPDF(smudge['sigma_time'], 0, 0)
                                timeNormalAtDist = normalPDF(smudge['sigma_time'], 0, abs(i-singularity['start']))
                                pxMovementMag = ((distNormalAtDist / distNormalAt0) * smudge['sigma_px_displ']) * (timeNormalAtDist / timeNormalAt0)

                                movement += np.array([pxMovementMag*singularity['oVector'][0], pxMovementMag*singularity['oVector'][1]]) #x,y

                            xyDest = [w,h] + movement
                            
                            if 0 <= xyDest[0] <= outputFrameDimensions[0]-1 and 0 <= xyDest[1] <= outputFrameDimensions[1]-1: 
                                subPixels[int(xyDest[0])][int(xyDest[1])].append([xyDest[0], xyDest[1], np.uint32(outputImage[h][w])]) #[h][w] because of inverted image storage
                
                    MAX_NEIGHBORHOOD_DIST = int(math.ceil(3 * smudge['sigma_neigh']))
                    for w in xrange(outputFrameDimensions[0]):
                        for h in xrange(outputFrameDimensions[1]):
                            cumulWeight = 0.
                            cumulWeighedIntens = 0.
                            for wn in xrange(w-MAX_NEIGHBORHOOD_DIST, w+MAX_NEIGHBORHOOD_DIST+1):
                                for hn in xrange(h-MAX_NEIGHBORHOOD_DIST, h+MAX_NEIGHBORHOOD_DIST+1):
                                    if not(0 <= wn < outputFrameDimensions[0] and 0 <= hn < outputFrameDimensions[1]):
                                        continue
    
                                    for subpx in subPixels[wn][hn]:
                                        fractW = subpx[0]
                                        fractH = subpx[1]
                                        intens = subpx[2]
    
                                        dist = np.linalg.norm(np.array([w,h]) - [fractW,fractH])
                                        #neighNormalAt0 = normalPDF(smudge['sigma_neigh'], 0, 0)
                                        neighNormalAtDist = normalPDF(smudge['sigma_neigh'], 0, dist)
                                        cumulWeighedIntens += intens * neighNormalAtDist
                                        cumulWeight += neighNormalAtDist
    
                            if cumulWeight > 0:
                                destPix[h][w] = np.uint8(cumulWeighedIntens/cumulWeight)

                    #show a circle for each singularity
                    if not args.batch:
                        for singularity in singularities:
                            cv2.circle(destPix, tuple(singularity['oPoint']), 3, (255,255,255)) #show singularities

                outputImage = destPix
                cv2.waitKey(1)

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

            if args.output:
                sink.write(modifiedImage)

        if not args.batch:
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