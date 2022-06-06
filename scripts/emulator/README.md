# emulator scripts

## Overview
This directory contains some scripts to support developing emulators on top of AbacusSummit simulations.  Typically, the workflow is that one wants to make a particular measurement
on many simulations, write those results to disk, and then do some kind of interpolation on the measurement.  `generate_cfs/` helps organize the first step, where one needs to repeat
a computation (the correlation function, in this case) on many sims.

## Usage
### generate_cfs
The `generate_cfs/` directory contains a script called `generate_cf.py` to evaluate the correlation function on a single
catalog.  The `launch_cori_slurm.sh` script then broadcasts that Python script over multiple sims/redshifts in the suite,
using [disBatch](https://github.com/flatironinstitute/disBatch/) to do the dynamic dispatch.  This script is particular
to NERSC's Cori platform, but the general pattern should be widely applicable.

