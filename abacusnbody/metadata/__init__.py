'''
Retrieve the cosmology and other code parameters associated with an
Abacus simulation.

Each set of simulations, like AbacusSummit, has a corresponding
repository of metadata. The simulation name will be used to infer
which repository to look in.
'''

from . import abacussummit

def get_meta(simname, redshift=None):
    '''
    Get the metadata associated with the given simulation.

    Parameters
    ----------
    simname : str
        The simulation name, like "AbacusSummit_base_ph000_c000".

    redshift : float or str
        The redshift

    Returns
    -------
    meta : dict
        The time-independent parameters and, if `redshift` is given,
        the time-dependent state values.
    '''

    if simname.startswith('AbacusSummit'):
        return abacussummit.get_meta(simname, redshift=redshift)

    raise ValueError(f'It is unknown what simulation set "{simname}" belongs to '
                      'based on the simulation name.')
