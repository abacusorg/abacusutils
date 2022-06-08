'''
Retrieve the cosmology and other code parameters associated with the
AbacusSummit simulations.
'''

import importlib.resources
import pickle

import asdf

metadata = None
metadata_fn = 'abacussummit_headers_compressed.asdf'

def get_meta(simname, redshift=None):
    '''
    Get the metadata associated with the given simulation.

    Parameters
    ----------
    simname : str
        The simulation name, like "AbacusSummit_base_ph000_c000".
        The "AbacusSummit_" prefix is optional.

    redshift : float or str
        The redshift

    Returns
    -------
    meta : dict
        The time-independent parameters and, if `redshift` is given,
        the time-dependent state values.
    '''

    if not simname.startswith('AbacusSummit_'):
        simname = 'AbacusSummit_' + simname
    
    global metadata
    if metadata == None:
        with importlib.resources.open_binary('abacusnbody.metadata', metadata_fn) as fp, asdf.open(fp) as af:
            if 'pickle' in af.tree:
                metadata = pickle.loads(af['pickle'][:])
            else:
                metadata = dict(af.tree)
    
    if simname not in metadata:
        raise ValueError(f'Simulation "{simname}" is not in metadata file "{metadata_fn}"')

    res = dict(metadata[simname]['param'])
    if 'CLASS_power_spectrum' in metadata[simname]:
        res['CLASS_power_spectrum'] = metadata[simname]['CLASS_power_spectrum']

    if redshift != None:
        if type(redshift) != str:
            redshift = f'z{redshift:.3f}'
        if not redshift.startswith('z'):
            redshift = 'z' + redshift
        if redshift not in metadata[simname]['state']:
            raise ValueError(f'Redshift {redshift} metadata not present for "{simname}" in metadata file "{metadata_fn}')
        res.update(metadata[simname]['state'][redshift])
    
    return res
