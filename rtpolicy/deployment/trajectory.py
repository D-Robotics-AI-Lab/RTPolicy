"""Piecewise-constant-jerk trajectory stitching in the inference layer."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class JointTrajectory:
    starts_at: float
    dt: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @property
    def ends_at(self):
        return self.starts_at + (len(self.position) - 1) * self.dt

    def sample(self, now):
        if now < self.starts_at or now > self.ends_at:
            return None
        index = min(int((now - self.starts_at) / self.dt), len(self.position) - 2)
        t = min(max(now - self.starts_at - index * self.dt, 0.0), self.dt)
        p, v, a = self.position[index], self.velocity[index], self.acceleration[index]
        jerk = (self.acceleration[index + 1] - a) / self.dt
        return p + v*t + a*t*t/2 + jerk*t**3/6, v + a*t + jerk*t*t/2, a + jerk*t


@dataclass(frozen=True)
class SplicedTrajectory:
    head: object
    tail: JointTrajectory

    def sample(self, now):
        if now >= self.tail.starts_at:
            return self.tail.sample(now)
        return self.head.sample(now) if self.head is not None else None


class PiecewiseJerkStitcher:
    """Track a time-aligned chunk with bounded velocity/acceleration/jerk.

    Decision variables per channel are position, velocity and acceleration at
    each knot. Initial p/v/a equal the active trajectory at the switch time.
    The two parameter groups follow TeleManipulation's arm/gripper defaults.
    """

    def __init__(self, dt=0.02, gripper_indices=()):
        try:
            import osqp
        except ImportError as error:
            raise ImportError('Streaming requires osqp; install the root requirements.txt') from error
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Trajectory knot interval must be positive')
        self.osqp, self.dt = osqp, float(dt)
        self.grippers = set(gripper_indices)
        self._solvers = {}

    def reset(self):
        self._solvers.clear()

    def _solver(self, count, grip):
        key = (count, grip)
        if key in self._solvers:
            return self._solvers[key]
        dt, n = self.dt, count
        vmax, amax, jmax, wa, wj = (10., 100., 2000., 1e-4, 1e-3) if grip else (3., 10., 200., .05, .05)
        eye = sparse.eye(n, format='csc')
        diff = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format='csc')
        track = np.ones(n)
        # Optimize v/vmax and a/amax: equivalent physical QP, well-scaled rows.
        cost_a = wa * eye + wj * (amax / (jmax * dt))**2 * (diff.T @ diff)
        P = sparse.block_diag([sparse.diags(2*track), sparse.csc_matrix((n,n)), 2*cost_a], format='csc')
        dynamics = sparse.lil_matrix((2*(n-1), 3*n))
        for k in range(n-1):
            dynamics[2*k, [k, k+1, n+k, 2*n+k, 2*n+k+1]] = [-1, 1, -dt*vmax, -dt**2*amax/3, -dt**2*amax/6]
            dynamics[2*k+1, [n+k, n+k+1, 2*n+k, 2*n+k+1]] = [-1, 1, -dt*amax/(2*vmax), -dt*amax/(2*vmax)]
        boundary = sparse.lil_matrix((3, 3*n))
        boundary[0, 0] = boundary[1, n] = boundary[2, 2*n] = 1
        A = sparse.vstack([dynamics, boundary, sparse.eye(3*n),
                           sparse.hstack([sparse.csc_matrix((n-1,2*n)), diff*amax/(dt*jmax)])], format='csc')
        lower = np.r_[np.zeros(2*(n-1)+3), np.full(n,-np.inf), np.full(2*n,-1.), np.full(n-1,-1.)]
        upper = np.r_[np.zeros(2*(n-1)+3), np.full(n,np.inf), np.full(2*n,1.), np.full(n-1,1.)]
        solver = self.osqp.OSQP()
        solver.setup(P=sparse.triu(P, format='csc'), q=np.zeros(3*n), A=A, l=lower, u=upper,
                     verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000,
                     warm_starting=True, polishing=True)
        value = solver, track, lower, upper, 2*(n-1), vmax, amax
        self._solvers[key] = value
        return value

    def stitch(self, actions, observed_at, switch_at, action_dt, active, measured):
        remaining = observed_at + (len(actions)-1)*action_dt - switch_at
        if remaining < 2*self.dt:
            return None
        count = max(2, int(remaining/self.dt + 0.5)) + 1
        times = switch_at + np.arange(count)*self.dt
        reference_times = observed_at + np.arange(len(actions))*action_dt
        boundary = active.sample(switch_at) if active is not None else None
        if boundary is None:
            boundary = (np.asarray(measured), np.zeros(actions.shape[1]), np.zeros(actions.shape[1]))
        p, v, a = (np.empty((count, actions.shape[1])) for _ in range(3))
        for column in range(actions.shape[1]):
            solver, weights, lower, upper, start, vmax, amax = self._solver(count, column in self.grippers)
            q = np.r_[-2*weights*np.interp(times, reference_times, actions[:,column]), np.zeros(2*count)]
            low, high = lower.copy(), upper.copy()
            low[start:start+3] = high[start:start+3] = [boundary[0][column], boundary[1][column]/vmax, boundary[2][column]/amax]
            solver.update(q=q, l=low, u=high)
            result = solver.solve(raise_error=False)
            if result.info.status_val not in (1, 2) or not np.isfinite(result.x).all():
                raise RuntimeError('Trajectory QP did not converge: '+result.info.status)
            p[:,column], velocity, acceleration = np.split(result.x, 3)
            v[:,column], a[:,column] = velocity*vmax, acceleration*amax
        return JointTrajectory(switch_at, self.dt, p, v, a)
