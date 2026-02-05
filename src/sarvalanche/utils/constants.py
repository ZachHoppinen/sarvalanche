import asf_search as asf
REQUIRED_ATTRS = {
    "source",
    "product",
    "units"
}

CANONICAL_DIMS_2D = ("y", "x")
CANONICAL_DIMS_3D = ("time", "y", "x")

SENTINEL1 = "SENTINEL-1"
OPERA_RTC = 'OPERA-RTC'
OPERA_RTC_STATIC = 'OPERA-RTC-STATIC'
OPERA_CSLC = 'OPERA-CSLC'
OPERA_CSLC_STATIC = 'OPERA-CSLC-STATIC'

RTC_FILETYPES = ['VV', 'VH', 'mask']
pols = ["VV", "VH"]

eps = 1e-6