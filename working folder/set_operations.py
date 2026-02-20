import numpy as np
import polytope as pc
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from tqdm import trange

def minkowski_sum(P: pc.Polytope, Q: pc.Polytope) -> pc.Polytope:
    Vp = pc.extreme(P)
    Vq = pc.extreme(Q)
    if Vp is None or Vq is None:
        raise ValueError("extreme() returned None (set may be empty, unbounded, or lower-dimensional).")
    Vsum = (Vp[:, None, :] + Vq[None, :, :]).reshape(-1, Vp.shape[1])
    return pc.reduce(pc.qhull(Vsum))

def pontryagin_difference(P: pc.Polytope, Q: pc.Polytope) -> pc.Polytope:
    # P ⊖ Q = { x | x + Q ⊆ P }
    # If P = {x | G x <= g}, then P ⊖ Q = {x | G x <= g - beta}, where beta_i = max_{q in Q} G_i q
    Gx, gx = P.A, np.ravel(P.b)

    Vq = pc.extreme(Q)
    if Vq is None:
        raise ValueError("extreme() returned None (Q may be empty, unbounded, or lower-dimensional).")

    beta = np.max(Vq @ Gx.T, axis=0)

    return pc.Polytope(Gx, gx - beta)

def is_subset(P, Q, eps=1e-9):
    # Check if Q is a subset of P (Q ⊆ P).
    Gx, gx = P.A, np.ravel(P.b)

    V = pc.extreme(Q)
    if V is None:
        return False
        
    return np.all(Gx @ V.T <= gx[:, None] + eps)


def point_in_set(P, x, eps=1e-9):
    # Check if a point x is in the set P (x ∈ P).
    Gx, gx = P.A, np.ravel(P.b)

    x = np.atleast_1d(x).flatten()

    return np.all(Gx @ x <= gx + eps)

def is_empty(P):
    # Check if the set P is empty.
    if P is None:
        return True

    if P.A is None or P.b is None:
        return True

    if P.A.size == 0 or P.b.size == 0:
        return True

    return False

def invariance_violation(X, Acl, W=None, lam=1.0):
    # Compute the invariance violation via containment.

    L = linear_map(Acl, X)
    if W is not None:
        L = minkowski_sum(L, W)

    V = pc.extreme(L)
    if V is None:
        return np.inf

    HX = X.A
    hX = np.ravel(X.b)

    viol = HX @ V.T - (lam * hX)[:, None]
    return float(np.max(viol))

def compute_max_invariant(p0, Acl, W=None, lam=1.0, max_iter=5000, eps=1e-9, viol_prog=False):
    # Compute the maximal positively invariant set via backward reachability.

    p0 = pc.reduce(p0)
    pn = p0

    if W is not None:
        W = pc.reduce(W)
        desc = "Computing RMPI"
    else:
        desc = "Computing MPI"

    for _ in trange(max_iter, desc=desc):
        if W is None:
            PreP = pc.Polytope(pn.A @ Acl, lam * np.ravel(pn.b))
        else:
            S = pc.Polytope(pn.A, lam * np.ravel(pn.b))
            S_tight = pc.reduce(pontryagin_difference(S, W))
            PreP = pc.Polytope(S_tight.A @ Acl, np.ravel(S_tight.b))

        pn = pc.reduce(pc.intersect(p0, PreP))

        if invariance_violation(pn, Acl, W, lam) <= eps:
            break

        if viol_prog:
            print(invariance_violation(pn, Acl, W, lam))

    return pn

def linear_map(A, P):
    # returns A P = {A x | x in P} using vertex mapping + convex hull
    V = pc.extreme(P)

    if V is None:
        raise ValueError("extreme() returned None (P may be empty, unbounded, or lower-dimensional).")

    VA = (A @ V.T).T
    return pc.reduce(pc.qhull(VA))

def compute_min_invariant(Acl, W, max_iter=5000, eps=1e-9, viol_prog=False):
    # Compute the minimal positively invariant set via forward reachability.

    W = pc.reduce(W)
    S = W

    for _ in trange(max_iter, desc='Computing mRPI'):

        S = minkowski_sum(W, linear_map(Acl, S))
        S = pc.reduce(S)
        
        viol = invariance_violation(S, Acl, W, lam=1.0)
        if viol <= eps:
            break
        if viol_prog:
            print(invariance_violation(S, Acl, W, lam=1.0))
    return S

def plot_poly(P, alpha=0.2, linewidth=2, color='blue', label=None, offset=None, axes=[0, 1]):
    # Plot a 2D projection of the polytope P using its vertices and convex hull.
    
    V = pc.extreme(P)

    if V is None or V.shape[0] < 3:
        return

    hull = ConvexHull(V[:, axes])
    Vh = V[hull.vertices]

    if offset is not None:
        offset = np.atleast_1d(offset).flatten()
        Vh = Vh + offset

    plt.fill(Vh[:, axes[0]], Vh[:, axes[1]], alpha=alpha, linewidth=linewidth, color=color, label=label)

def linear_image(P: pc.Polytope, K, d=None) -> pc.Polytope:
    # Compute the linear image of a polytope P under the map x -> K x + d.
    
    V = pc.extreme(P)
    if V is None:
        raise ValueError("extreme() returned None (empty/unbounded/lower-dimensional).")

    K = np.asarray(K, dtype=float)
    if K.ndim == 1:
        K = K.reshape(1, -1)

    n = V.shape[1]
    m, nK = K.shape
    if nK != n:
        raise ValueError(f"K has {nK} columns but P lives in R^{n}.")

    if d is None:
        d = np.zeros(m)
    d = np.asarray(d, dtype=float).reshape(m)

    # Map vertices: (Nv, n) @ (n, m) -> (Nv, m)
    W = V @ K.T
    W = W + d  # broadcast

    # Special-case 1D output: build interval polytope directly (more reliable than qhull in 1D)
    if m == 1:
        wmin = float(np.min(W))
        wmax = float(np.max(W))
        A = np.array([[1.0], [-1.0]])
        b = np.array([wmax, -wmin])
        return pc.reduce(pc.Polytope(A, b))

    # General case: convex hull of mapped points
    return pc.reduce(pc.qhull(W))
