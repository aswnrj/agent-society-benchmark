from mosstool.map.osm import RoadNet, Building
from mosstool.map.builder import Builder
from mosstool.util.format_converter import dict2pb
from mosstool.type import Map

import geojson.feature as _gjf

_orig_fc_init = _gjf.FeatureCollection.__init__

def _patched_fc_init(self, features, **extra):
    if isinstance(features, dict) and "features" in features:
        features = features["features"]
    _orig_fc_init(self, features, **extra)


_gjf.FeatureCollection.__init__ = _patched_fc_init

min_lat, max_lat = 39.78, 39.92
min_lon, max_lon = 116.32, 116.40

projstr = f"+proj=tmerc +lat_0={(min_lat+max_lat)/2} +lon_0={(min_lon+max_lon)/2}"

rn = RoadNet(proj_str=projstr, max_latitude=max_lat, min_latitude=min_lat, max_longitude=max_lon, min_longitude=min_lon, proxies=None)
roadnet = rn.create_road_net()

bld = Building(proj_str=projstr, max_latitude=max_lat, min_latitude=min_lat, max_longitude=max_lon, min_longitude=min_lon, proxies=None)
aois = bld.create_building()

m = Builder(net=roadnet, aois=aois, pois=[], proj_str=projstr).build("bench_map")

with open("map.pb", "wb") as f:
    f.write(dict2pb(m, Map()).SerializeToString())
print("wrote map.pb")