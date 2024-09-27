"""
Retrieve the cosmology and other code parameters associated with the
AbacusSummit simulations.
"""

import sys

if sys.version_info >= (3, 9):
    import importlib.resources as resources
else:
    import importlib_resources as resources

import asdf
import msgpack

metadata = None
metadata_fns = [
    'abacussummit_headers_compressed.asdf',
    'abacusdesi2_headers_compressed.asdf',
]


def get_meta(simname, redshift=None):
    """
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
    """

    global metadata
    if metadata is None:
        metadata = {}
        for metadata_fn in metadata_fns:
            with asdf.open(resources.files('abacusnbody.metadata') / metadata_fn) as af:
                af_tree = dict(af.tree)
                del af_tree['asdf_library'], af_tree['history']
                for sim in af_tree:
                    metadata[sim] = {}
                    metadata[sim]['param'] = msgpack.loads(
                        af_tree[sim]['param'].data, strict_map_key=False
                    )
                    metadata[sim]['state'] = msgpack.loads(
                        af_tree[sim]['state'].data, strict_map_key=False
                    )
                    if 'CLASS_power_spectrum' in af_tree[sim]:
                        metadata[sim]['CLASS_power_spectrum'] = af_tree[sim][
                            'CLASS_power_spectrum'
                        ]
    if simname not in metadata:
        raise ValueError(
            f'Simulation "{simname}" is not in metadata files "{metadata_fns}"'
        )

    res = dict(metadata[simname]['param'])
    if 'CLASS_power_spectrum' in metadata[simname]:
        res['CLASS_power_spectrum'] = metadata[simname]['CLASS_power_spectrum']

    if redshift is not None:
        if not isinstance(redshift, str):
            redshift = f'z{redshift:.3f}'
        if not redshift.startswith('z'):
            redshift = 'z' + redshift
        if redshift not in metadata[simname]['state']:
            raise ValueError(
                f'Redshift {redshift} metadata not present for "{simname}" in metadata files "{metadata_fns}'
            )
        res.update(metadata[simname]['state'][redshift])

    return res
