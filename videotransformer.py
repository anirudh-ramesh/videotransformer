#!/usr/bin/env python2

import cv2
import sys
import argparse


def nop(*args):
    pass

def main():
    parser = argparse.ArgumentParser(description='Process video files')
    parser.add_argument('vid_in', help='Input video file')
    parser.add_argument('--vid_out',  default='', help='Output video file')
    parser.add_argument('--size', default='0x0', help='WxH')
    parser.add_argument('--grey', action='store_true', help='Convert to greyscale')
    parser.add_argument('--mirrorh', action='store_true', help='Mirror image horizontally')
    parser.add_argument('--mirrorv', action='store_true', help='Mirror image vertically')
    parser.add_argument('--speedup', type=int, default=1, choices=xrange(1,5), help='Speed-up playback (integer factor)')
    args = parser.parse_args()
    print repr(args)

    try:
        outputFrameWidth = int(args.size.split('x')[0])
        outputFrameHeight = int(args.size.split('x')[1])
    except ValueError:
        outputFrameWidth = 0
        outputFrameHeight = 0

    cap = cv2.VideoCapture(args.vid_in)
    if not cap.isOpened():
        sys.stderr.write("Couldn't open video file.\n")
        exit(1)

    FPS = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    NB_FRAMES = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    INPUT_FRAME_WIDTH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    INPUT_FRAME_HEIGHT = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    sys.stderr.write('In: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, INPUT_FRAME_WIDTH, INPUT_FRAME_HEIGHT))

    outputFrameHeight = outputFrameHeight if outputFrameHeight > 0 else INPUT_FRAME_HEIGHT
    outputFrameWidth = outputFrameWidth if outputFrameWidth > 0 else INPUT_FRAME_WIDTH
    sys.stderr.write('Out: fps={} frames={} {}x{}\n'.format(FPS, NB_FRAMES, outputFrameWidth, outputFrameHeight))

    if not FPS*NB_FRAMES*INPUT_FRAME_WIDTH*INPUT_FRAME_HEIGHT:
        sys.stderr.write("Video doesn't appear to be supported.\n")
        exit(1)

    if args.vid_out:
        sink = cv2.VideoWriter(args.vid_out, cv2.cv.FOURCC('M', 'J', 'P', 'G'), FPS, (outputFrameWidth,outputFrameHeight), isColor=not args.grey)
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
        if INPUT_FRAME_HEIGHT != outputFrameHeight or INPUT_FRAME_WIDTH != outputFrameWidth:
            outputImage = cv2.resize(outputImage, (outputFrameWidth, outputFrameHeight))

        #
        if args.grey:
            outputImage = cv2.cvtColor(outputImage, cv2.cv.CV_BGR2GRAY)

        #
        if args.mirrorh and args.mirrorv:
            outputImage = cv2.flip(outputImage, -1)
        elif args.mirrorh:
            outputImage = cv2.flip(outputImage, 1)
        elif args.mirrorv:
            outputImage = cv2.flip(outputImage, 0)

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
