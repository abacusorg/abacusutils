# Stop astropy from trying to download time data; nodes on some clusters are not allowed to access the internet directly
from astropy.utils import iers

iers.conf.auto_download = False
