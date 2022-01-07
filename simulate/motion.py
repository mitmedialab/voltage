import math
import random


DAMPING = 0.5
ANGLE_PERTURBATION = 0.2
SCALE = 20.0
TIME_STEP = 0.1


def synthesize_motion(time_frames):
    """
    Synthesize motion to be introduced to simulated microscope video.
    To keep motion vectors from going too far away from the origin,
    the force is exerted toward the origin with some angle perturbation.
    The motion is simulated by double integration of acceleration with
    damping applied to velocity.

    Parameters
    ----------
    time_frames : int
        The number of time frames for which motion vectors to be synthesized.

    Returns
    -------
    Xs : list of float
        X coordinates of motion vectors
    Ys : list of float
        Y coordinates of motion vectors

    """

    pi = math.pi
    x = 0
    y = 0
    vx = 0
    vy = 0
    Xs = []
    Ys = []

    for i in range(time_frames):

        Xs.append(x)
        Ys.append(y)

        if(x == 0 and y == 0):
            angle = random.uniform(-pi, +pi) # pick direction randomly
        else: # set the angle so the motion tends to stick around the origin
            angle = math.atan2(y, x) # current angle from the origin
            angle += pi # opposite direction to return to the origin
            angle += random.uniform(-pi, +pi) * ANGLE_PERTURBATION

        mag = random.random() * SCALE
        ax = mag * math.cos(angle) - DAMPING * vx
        ay = mag * math.sin(angle) - DAMPING * vy

        vx += ax * TIME_STEP
        vy += ay * TIME_STEP
        x += vx * TIME_STEP
        y += vy * TIME_STEP

    return Xs, Ys


def _test():
    Xs, Ys = synthesize_motion(1000)
    import matplotlib.pyplot as plt
    plt.plot(Xs, Ys)
    plt.show()
