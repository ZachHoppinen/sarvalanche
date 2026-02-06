import asf_search as asf
from sarvalanche.utils.validation import validate_urls

def find_asf_urls(aoi,
                start_date,
                stop_date,
                product_type = asf.PRODUCT_TYPE.RTC,
                path_number = None,
                burst_id = None,
                direction = None,
                frame = None):

    asf_results = asf.geo_search(
        intersectsWith=aoi.wkt,
        start=start_date,
        end=stop_date,
        processingLevel=product_type,
        relativeOrbit=path_number,
        relativeBurstID=burst_id,
        flightDirection=direction,
        frame = frame)

    urls =  asf_results.find_urls()

    if product_type == asf.PRODUCT_TYPE.RTC:
        # Apply extension filtering only for RTC products
        extensions=("_VV.tif", "_VH.tif", "_mask.tif")
        urls = [u for u in urls if any(u.endswith(ext) for ext in extensions)]

    elif product_type == asf.PRODUCT_TYPE.CSLC:
        extension = '.h5'
        urls = [u for u in urls if u.endswith(extension)]

    elif product_type == asf.PRODUCT_TYPE.RTC_STATIC:
        extension = 'local_incidence_angle.tif'
        urls = [u for u in urls if u.endswith(extension)]

    return validate_urls(urls)

import earthaccess
from itertools import chain

def find_earthaccess_urls(aoi, start_date, stop_date, short_name = "WUS_UCLA_SR"):
    auth = earthaccess.login()
    results = earthaccess.search_data(
        short_name = short_name,
        cloud_hosted = True,
        bounding_box = aoi.bounds,
        temporal = (start_date, stop_date),
    )
    urls = [r.data_links() for r in results]
    # flatten list
    urls = list(chain.from_iterable(urls))
    if short_name == "WUS_UCLA_SR":
        # get only swe results
        urls = [u for u in urls  if 'SWE_SCA_POST.nc' in u]
    return validate_urls(urls)
