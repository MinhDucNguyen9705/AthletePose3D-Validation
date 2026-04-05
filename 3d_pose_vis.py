import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine
    (8, 11), (11, 12), (12, 13),       # left arm
    (8, 14), (14, 15), (15, 16)        # right arm
]

KEYPOINTS = [
    'Root', 'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle',
    'Belly', 'Neck', 'Nose', 'Head', 
    'RShoulder', 'RElbow', 'RHand', 'LShoulder', 'LElbow', 'LHand'
]

pose3d = np.load("AthletePose3D/data/train_set/S1/Axel_1_cam_1_h36m.npy")  # (150, 17, 3)
print(pose3d.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(t):
    ax.cla()
    pts = pose3d[t]

    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    ax.scatter(xs, ys, zs)

    for i, j in H36M_EDGES:
        ax.plot([xs[i], xs[j]],
                [ys[i], ys[j]],
                [zs[i], zs[j]])
        # plot the id of points

    # for i in range(pts.shape[0]):
    #     ax.text(xs[i], ys[i], zs[i], KEYPOINTS[i], color='red')

    center = pts.mean(axis=0)
    r = np.max(np.ptp(pts, axis=0)) / 2

    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)

ani = FuncAnimation(fig, update, frames=len(pose3d), interval=50)
plt.show()