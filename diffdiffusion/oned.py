import numpy as np
from jax import config
config.update("jax_enable_x64", True)

import lineax as lx
from jax import vmap, numpy as jnp, jit
from functools import partial
import diffrax
import equinox as eqx


def get_analytical_1d(diff_coeff, t, ax):
    """
    Returns the analytical solution of the diffusion equation to a normalized gaussian initial condition
    
    We use this to generate the initial condition, $t=1000$ and solution at a later time, $t > 1000$
    
    Args: 
    diff_coeff: diffusion coefficient
    t: time
    ax: axis
    
    Returns:
    solution array at time t that is (ax.size, ) in shape
    
    """
    
    return 1/np.sqrt(4*np.pi*diff_coeff*t)*np.exp(-ax**2./4/diff_coeff/t)

def solve_diff_1d(dt, finp, diff_coeff):
    """
    Solves 1 diffusion timestep
    
    NB: v is hardcoded here because of global variables in Jupyter Notebooks.
    Do not do this in a production environment 
    
    Args:
    dt: timestep
    finp: input array
    diff_coeff: diffusion coefficient
    
    Returns:
    solution to the diffusion equation
    
    """
    coeff = -dt * diff_coeff / dv**2.
    diag = 1-np.concatenate([[1.], 2*np.ones_like(v[1:-1]), [1.]])*coeff
    lower_diag = np.ones_like(v[1:])*coeff
    upper_diag = np.ones_like(v[1:])*coeff
    operator = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
    solution = lx.linear_solve(operator, finp, solver=lx.Tridiagonal())
    return solution.value

class VectorField(eqx.Module):
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """
    v: jnp.ndarray
    dt: float
    kappa: float
    
    def __init__(self, v: jnp.ndarray, dt: float, kappa: float):
        super().__init__()
        self.v = v
        self.dt = dt
        self.kappa = kappa

    def __call__(self, t: float, y: jnp.ndarray, args):
        return solve_diff_1d(self.dt, y, self.kappa)
        