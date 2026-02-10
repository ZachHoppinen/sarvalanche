# weighting functions
from sarvalanche.weights.local_resolution import get_local_resolution_weights
from sarvalanche.weights.temporal import get_temporal_weights


def get_static_weights(ds, avalanche_date):

    ds['w_resolution'] = get_local_resolution_weights(ds['anf'])
    ds['w_temporal'] = get_temporal_weights(ds['time'], avalanche_date)

    for d in ['w_resolution', 'w_temporal']:
        ds[d].attrs = {'source': 'sarvalance', 'units': '1', 'product': 'weight'}

    return ds
