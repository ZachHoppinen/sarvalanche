import asf_search as asf
REQUIRED_ATTRS = {
    "sensor",
    "product",
    "crs",
    "source_urls",
    "units"
}

CANONICAL_DIMS_2D = ("y", "x")
CANONICAL_DIMS_3D = ("time", "y", "x")

SENTINEL1 = "SENTINEL-1"
OPERA_RTC = asf.PRODUCT_TYPE.RTC
OPERA_RTC_STATIC = asf.PRODUCT_TYPE.RTC_STATIC
OPERA_CSLC = asf.PRODUCT_TYPE.CSLC
OPERA_CSLC_STATIC = asf.PRODUCT_TYPE.CSLC_STATIC
