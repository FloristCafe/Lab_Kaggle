"""Compare all float32 drift times with the pinned JAX operator."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp
import numpy as np

from agents.mappo.shared.physics import SPEEDS, drift_offset


t=jnp.arange(-1,505,dtype=jnp.float32)
for speed in SPEEDS:
    phase=(t*jnp.abs(jnp.float32(speed)))%1
    phase=np.asarray(phase)
    expected=np.concatenate(([0],np.cumsum(phase[:-1]>phase[1:]))) * np.sign(speed)
    actual=[drift_offset(speed,i) for i in range(506)]
    assert np.array_equal(actual,expected), (speed,np.flatnonzero(actual!=expected))
print("PASS 8 x 506 observation indices")
