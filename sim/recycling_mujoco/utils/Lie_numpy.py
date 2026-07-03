import numpy as np

# WE WILL FIX THIS LATER (SIMILAR TO THE TORCH VERSION)
# BATCHWISE, FORMAL, EFFICIENT (NO FOR LOOP, NO EXTERNAL LIBRARY, REFER TO THE MR CODE)

def quat2SO3(quaternion):
    # Input: quaternion with shape (K, 4) and order (x, y, z, w)
    assert quaternion.shape[1] == 4

    K = quaternion.shape[0]
    R = np.zeros((K, 3, 3), dtype=quaternion.dtype)

    # A unit quaternion is q = w + xi + yj + zk
    x = quaternion[:, 0]
    y = quaternion[:, 1]
    z = quaternion[:, 2]
    w = quaternion[:, 3]

    xx = x**2
    yy = y**2
    zz = z**2
    ww = w**2
    n = ww + xx + yy + zz  # shape (K,)
    s = np.zeros(K, dtype=quaternion.dtype)
    nonzero = (n != 0)
    s[nonzero] = 2 / n[nonzero]

    xy = s * x * y
    xz = s * x * z
    yz = s * y * z
    xw = s * x * w
    yw = s * y * w
    zw = s * z * w

    xx = s * xx
    yy = s * yy
    zz = s * zz

    idxs = np.arange(K)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

# skew
def skew(w):
    if len(w) == 3:
        W = np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])
    if len(w) == 6:
        W = np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])
        W = np.vstack([
            np.concatenate([W, w[3:].reshape(3,1)], axis=1), np.array([0,0,0,0])
        ])
    return W

def invskew(W):
    if len(W) == 3:
        w = np.array([-W[1,2], W[0,2], -W[0,1]])
    if len(W) == 4:
        w = np.array([-W[1,2], W[0,2], -W[0,1]])
        w = np.hstack([w, W[:3, 3]])
    return w
    
# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    if w.shape == (3,3):
        w = invskew(w)
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)
    return R

# SE3 exponential
def exp_se3(S):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    if S.shape == (4,4):
        S = invskew(S)
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v.reshape(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        P = np.eye(3) + (1 - cw) * np.power(wnorm_inv,2) * W + (wnorm - sw) * np.power(wnorm_inv,3) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = exp_so3(w)
        T[0:3,3] = P.dot(v).reshape(3)
    return T

def clipping(x, low=-1.0, high=1.0):
    eps = 1e-6
    if x <= low:
        x = low + eps
    elif x >= high:
        x = high - eps
    return x

# SO3 log
def log_SO3(R):
    angle_threshold = 1e-6
    trace = np.trace(R)
    theta = np.arccos(clipping((trace - 1) / 2))
    if np.abs(trace + 1) >= angle_threshold:
        skew_w = (R - R.transpose()) / (2 * np.sin(theta)) * theta
    elif np.abs(trace - 3) < 1e-10:
        skew_w = np.zeros((3,3))
    elif np.abs(trace + 1) < angle_threshold:
        if not(np.abs(R[2,2] + 1) < angle_threshold):
            r = R[2, 2]
            w = R[:, 2]
            w[2] += 1
        elif not(np.abs(R[1,1] + 1) < angle_threshold):
            r = R[1,1]
            w = R[:,1]
            w[1] += 1
        elif not(np.abs(R[0,0] + 1) < angle_threshold):
            r = R[0,0]
            w = R[:,0]
            w[0] += 1
        else:
            # print(f'ERROR: should be fixed.')
            NotImplementedError
        skew_w = skew(np.pi / np.sqrt(2 * (1 + r)) * w)
    return skew_w

# # SE3 log
# def log_SE3(T):
#     angle_threshold = 1e-6
#     R = T[:3,:3]
#     trace = np.trace(R)
#     skew_S = np.zeros((4,4))
#     if np.abs(trace-3) < angle_threshold:
#         skew_S[:3, 3] = T[:3, 3]
#     if np.abs(trace-3) >= angle_threshold:
#         skew_w = log_SO3(R)
#         theta = np.arccos(clipping(0.5*(trace-1)))
#         wmat = skew_w / theta
#         identity = np.eye(3)
#         invG = (1/theta) * identity - 0.5 * wmat + (1/theta - 0.5/np.tan(0.5*theta)) * wmat@wmat
#         skew_S[:3, :3] = skew_w
#         skew_S[:3, 3] = (theta * (invG@T[:3, 3:4])).reshape(3)
#     return skew_S

# Adjoint of SE3
def Adjoint_SE3(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    skewp = skew(p)
    
    AdT = np.eye(6)
    AdT[0:3,0:3] = R
    AdT[0:3,3:6] = np.zeros((3,3))
    AdT[3:6,0:3] = skewp.dot(R)
    AdT[3:6,3:6] = R
    return AdT

# small adjoint of se3
def adjoint_se3(S):
    w = S[0:3]
    v = S[3:6]
    skeww = skew(w)
    skewv = skew(v)
    
    adS = np.zeros((6,6))
    adS[0:3,0:3] = skeww
    adS[3:6,0:3] = skewv
    adS[3:6,3:6] = skeww
    return adS

# Joint and theta to SO3
def Joint_to_SO3(w, theta):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    cw = np.cos(theta)
    sw = np.sin(theta)
    W = skew(w)
    R = np.eye(3) + sw * W + (1 - cw) * W.dot(W)
    return R

# Screw and theta to SE3
def Screw_to_SE3(S, theta):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))    
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v * theta
    else:
        cw = np.cos(theta)
        sw = np.sin(theta)
        W = skew(w)
        P = theta * np.eye(3) + (1 - cw) * W + (theta - sw) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = Joint_to_SO3(w, theta)
        T[0:3,3] = P.dot(v).reshape(3)
    return T

def forward_kinematics(jointPos, S_screw, initialEEFrame):
    LinkFrames_from_base = []
    temp = np.eye(4)
    for q, S in zip(jointPos, S_screw):
        # temp = temp@exp_se3(S*q)
        temp = np.matmul(temp, exp_se3(S*q))
        LinkFrames_from_base.append(temp.reshape(1,4,4))
    LinkFrames_from_base = np.concatenate(LinkFrames_from_base, axis=0)
    EEFrame = np.matmul(LinkFrames_from_base[-1], initialEEFrame)
    return LinkFrames_from_base, EEFrame

def get_SpaceJacobian(S_screw, LinkFrames_from_base):
    SpaceJacobian = []
    SpaceJacobian.append(S_screw[0].reshape(6,1))
    for T, S in zip(LinkFrames_from_base[:-1], S_screw[1:]):
        SpaceJacobian.append(
            np.matmul(Adjoint_SE3(T),S).reshape(6,1)
        )
    SpaceJacobian = np.concatenate(SpaceJacobian, axis=1)
    return SpaceJacobian

def inverse_kinematics():
    pass


def invSE3(SE3):
    """
    Compute the inverse of a batch of SE(3) matrices, without matrix inversion.
    Args:
        SE3 (np.ndarray): shape (nBatch, 4, 4), batch of SE(3) matrices.
    
    Returns:
        np.ndarray: shape (nBatch, 4, 4), batch of inverse SE(3) matrices.
    """
    is_batch = True 
    # Support both batch and non-batch inputs
    if len(SE3.shape) == 2:
        SE3 = np.expand_dims(SE3, 0)
        is_batch = False
        
    nBatch = len(SE3)
    R, p = SE3[:, :3, :3], np.expand_dims(SE3[:, :3, 3], -1)
    invSE3_ = np.zeros((nBatch, 4, 4))
    inv_R = R.transpose(0, 2, 1) 
    invSE3_[:, :3, :3] = inv_R
    invSE3_[:, :3, 3] = - np.squeeze(np.matmul(inv_R, p), -1) # inv_R @ p is not supported for older numpy versions
    invSE3_[:, 3, 3] = 1
    
    if not is_batch:
        invSE3_ = np.squeeze(invSE3_, 0)
    return invSE3_
