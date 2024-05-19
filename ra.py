from read_calib import read_json_calib, k_matrix, d_coeff
import cv2 as cv
import numpy as np
from typing import List, Tuple
import os
from enum import Enum

CALIB_FILE = 'calib_logi_c270_hd_webcam__046d_0825__1280.json'
PATH = os.path.join(os.path.dirname(__file__), CALIB_FILE)


def homog(x):
    """Converts a set of ordinary points into homogeneous coordinates."""
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

def inhomog(x):
    """Converts a set of homogeneous points into ordinary coordinates."""
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]


def htrans(h,x):
    """Applies a homogeneous transformation h to a set of points x."""
    return inhomog(homog(x) @ h.T)

def rmsreproj(view, model, transf):
    """Computes the RMS reprojection error of the model with respect to the view."""
    err = view - htrans(transf,model)
    return np.sqrt(np.mean(err.flatten()**2))

def get_M(reference: np.ndarray, view: np.ndarray) -> (
        Tuple[float, np.ndarray]|
        Tuple[float, None]):
    """
    Return the transformation matrix M from the reference to the view."""

    K = k_matrix(calib)
    D = d_coeff(calib)
    ok,rvec,tvec =  cv.solvePnP(reference, view, K, D)
    if not ok:
        return 1e6, None
    R,_ = cv.Rodrigues(rvec)
    M = K @ np.hstack((R, tvec))
    rms = rmsreproj(view,reference,M)
    return rms, M


def orientation(x):
    """
    Compute the orientation of a contour."""
    return cv.contourArea(x.astype(np.float32),oriented=True)

def redondez(c):
    """
    Compute the orientation and roundness of a contour."""
    p = cv.arcLength(c.astype(np.float32),closed=True)
    oa = orientation(c)
    if p>0:
        return oa, 100*4*np.pi*abs(oa)/p**2
    else:
        return 0,0

def boundingBox(c):
    """
    Compute the bounding box of a contour."""
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

def internal(c,h,w):
    """
    Check that the contour is inside the image."""
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1>1 and x2 < w-2 and y1 > 1 and y2 < h-2

def redu(c,eps=0.5):
    """Reduce the number of nodes in a contour."""
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    """
    Attempts to detect polygons with n sides
    
    Args:
    cs: list of contours
    n: number of sides
    prec: approximation precision
    
    Returns:
    List of contours that are polygons with n sides"""

    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if r.shape[0] == n ]

def extractContours(g, minarea=10, minredon=25, reduprec=1) -> List[np.ndarray]:
    """
    Extract contours from a grayscale image.
    
    Args:
    g: grayscale image
    minarea: minimum area of the contour
    minredon: minimum redondez of the contour
    reduprec: approximation precision
    
    Returns:
    List of contours that satisfy the conditions.
    """

    #gt = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,-10)
    ret, gt = cv.threshold(g,189,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    contours = cv.findContours(gt, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]

    h,w = g.shape

    tharea = (min(h,w)*minarea/100.)**2 

    def good(c):
        oa,r = redondez(c)
        black = oa > 0 # and positive orientation
        return black and abs(oa) >= tharea and r > minredon

    ok = [redu(c.reshape(-1,2),reduprec) for c in contours if good(c)]
    return [ c for c in ok if internal(c,h,w) ]

def shcont(c, frame:cv.typing.MatLike, color=(255,0,0), nodes=True):
    """
    Show contour on the frame using OpenCV.
    
    Args:
    c: contour
    frame: image frame
    color: color of the contour (default: blue)
    nodes: whether to show nodes (default: True)
    """
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    pts = np.column_stack((x, y)).reshape((-1, 1, 2)).astype(np.int32)
    cv.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    if nodes:
        for i in range(len(x)):
            cv.circle(frame, (x[i], y[i]), radius=4, color=color, thickness=-1)
    
    return frame

def rots(c: np.ndarray) -> List[np.ndarray]:
    """
    Generate all the rotations of a contour."""
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(view: np.ndarray, model: np.ndarray) -> Tuple[float, np.ndarray]|Tuple[float, None]:
    """
    Return the best pose of the model with respect to the view.

    Args:
    view: view contour
    model: model contour

    Returns:
    Tuple with the RMS error and the transformation matrix M.
    """

    poses = [ get_M(model, v.astype(float)) for v in rots(view) ]
    return sorted(poses,key=lambda p: p[0])[0]

MOVEMENTS = Enum('MOVEMENTS', ['UP', 'DOWN', 'LEFT', 'RIGHT'])

def move(cube: np.ndarray, direction: MOVEMENTS, step: int = 1) -> np.ndarray:
    """
    Move the cube in a direction.

    Args:
    cube: cube
    direction: direction
    step: step

    Returns:
    Moved cube.
    """
    if direction == MOVEMENTS.UP:
        return cube + np.array([0, step, 0])
    elif direction == MOVEMENTS.DOWN:
        return cube + np.array([0, -step, 0])
    elif direction == MOVEMENTS.LEFT:
        return cube + np.array([-step, 0, 0])
    elif direction == MOVEMENTS.RIGHT:
        return cube + np.array([step, 0, 0])

def move_to_click(cube: np.ndarray, click: Tuple[int, int], M: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Move the cube to the click position.

    Args:
    cube: cube
    click: click position

    Returns:
    Moved cube.
    """

    H, _ = cv.findHomography(htrans(M, ref), ref)
    click_h = htrans(H, click)
    click_h = np.array([click_h[0], click_h[1], 0])
    return cube + click_h

if __name__ == '__main__':

    from umucv.stream import autoStream
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', type=str, default=PATH, help='Calibration file')
    args, _ = parser.parse_known_args()
    PATH = os.path.join(args.calib)
    calib = read_json_calib(PATH)

    REF_POINTS = np.array([[0. , 0. , 0. ],
       [0. , 1. , 0. ],
       [0.5, 1. , 0. ],
       [0.5, 0.5, 0. ],
       [1. , 0.5, 0. ],
       [1. , 0. , 0. ]])
    CUBE = np.array([
        [0,0,0],
        [1,0,0],
        [1,1,0],
        [0,1,0],
        [0,0,0],

        [0,0,1],
        [1,0,1],
        [1,1,1],
        [0,1,1],
        [0,0,1],

        [1,0,1],
        [1,0,0],
        [1,1,0],
        [1,1,1],
        [0,1,1],
        [0,1,0]])/2
    cube = CUBE
    
    click: Tuple[int, int] | None = None
    def onMouse(event, x, y, flags, param):
        global click
        if event == cv.EVENT_LBUTTONDOWN:
            click = (x, y)

    cv.namedWindow('frame')
    cv.setMouseCallback('frame', onMouse)


    for key,frame in autoStream():
        g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        conts = extractContours(g, reduprec=3)
        goods = polygons(conts,6)
        
        for c in conts:
            frame = shcont(c,frame, nodes=False)
        for g in goods:
            rms,M = bestPose(g,REF_POINTS)
            
            if M is not None:
                frame = shcont(g,frame, color=(0,0,255))
                frame = shcont( htrans(M, cube),frame, nodes=False,color=(0,255,0))
                if click is not None:
                    cube = move_to_click(CUBE, click, M, REF_POINTS)
                    click = None

        if key == ord('w'):
            cube = move(cube, MOVEMENTS.UP)
        elif key == ord('s'):
            cube = move(cube, MOVEMENTS.DOWN)
        elif key == ord('a'):
            cube = move(cube, MOVEMENTS.LEFT)
        elif key == ord('d'):
            cube = move(cube, MOVEMENTS.RIGHT)
        
        if key == ord('q'):
            break
        
        cv.imshow('frame',frame)
    
    

