# weighting functions
import logging

from sarvalanche.weights.local_resolution import get_local_resolution_weights
from sarvalanche.weights.temporal import get_temporal_weights

log = logging.getLogger(__name__)


def get_static_weights(ds, avalanche_date, temporal_decay_factor=6):
    log.info("Computing resolution weights")
    ds['w_resolution'] = get_local_resolution_weights(ds['anf'])

    log.info("Computing temporal weights (tau_days=%s)", temporal_decay_factor)
    ds['w_temporal'] = get_temporal_weights(ds['time'], avalanche_date, tau_days=temporal_decay_factor)

    for d in ['w_resolution', 'w_temporal']:
        ds[d].attrs = {'source': 'sarvalance', 'units': '1', 'product': 'weight'}

    return ds
