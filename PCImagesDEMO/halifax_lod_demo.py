"""
Halifax / Moncton Building LOD 3.1 demo pipeline.

address -> geocode -> footprint -> lidar crop -> street images -> roofer (LOD 2.2)
        -> image-detected openings back-projected onto walls -> LOD 3.1 CityJSON
        -> manifest.json

See plan.md for the full design rationale. Endpoints marked TODO(verify) must be
checked against live docs before this is used for anything beyond a demo run --
do not trust the URLs/params below without confirming them yourself.

Requires: requests, pyproj, shapely, numpy, torch, torchvision, transformers, pillow
Requires on PATH: pdal, roofer

Opening detection uses Grounding DINO (open-vocabulary object detector,
prompted with "window."/"door.") via Hugging Face transformers -- CUDA is
used automatically if available, falls back to CPU otherwise. Detections are
cross-checked against Depth Anything V2 (metric outdoor checkpoint): a real
window correctly detected on a distant, unrelated background building will
still satisfy the ray-plane wall intersection (monocular back-projection has
no depth signal of its own), so an independent per-pixel depth estimate is
the only thing that catches it. Model weights download on first use and are
cached under ~/.cache/huggingface.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, mapping, shape

USER_AGENT = "halifax-lod-demo/0.1 (contact: media@cims.carleton.ca)"

# ---------------------------------------------------------------------------
# Config -- TODO(verify) items called out in plan.md section 7.
# ---------------------------------------------------------------------------

CONFIG = {
    # Nominatim: stable, documented, no key needed. Respect the 1 req/sec rate limit.
    "nominatim_url": "https://nominatim.openstreetmap.org/search",

    # Confirmed live: HRM's "Buildings" hosted feature service, layer 0
    # ("Building Polygons", esriGeometryPolygon) -- found via the ArcGIS Hub
    # item page (https://data-hrm.hub.arcgis.com/maps/1f73c9ad861a4fd9998c097d3544c08e)
    # and its content/items/<id>?f=json metadata, not guessed.
    "hrm_footprint_layer_url": (
        "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/"
        "Buildings/FeatureServer/0/query"
    ),

    # Confirmed live: GeoNOVA's "LIDAR Tiles" layer (layer 2) on the
    # ELEV_LIDAR_ELEVATION_UT83 MapServer, found via the DataLocator Elevation
    # Explorer page (https://nsgi.novascotia.ca/datalocator/elevation/). Its
    # extent reports spatialReference wkid 2038 / latestWkid 2961, confirming
    # the EPSG guess below. TILENUMB + PRODNAME give the tile grid cell and
    # the LAZ filename ("438_4946_201901.laz"), but there is NO static
    # per-tile download URL field -- the only "download" field on the parent
    # project record just points back to the DataLocator UI. Actual bulk LAZ
    # retrieval goes through that portal's self-serve download flow, which
    # still needs to be reverse-engineered from the browser network tab
    # (TODO(verify) -- unresolved, not fabricated).
    "geonova_lidar_tiles_url": (
        "https://nsgiwa.novascotia.ca/arcgis/rest/services/ELEV/"
        "ELEV_LIDAR_ELEVATION_UT83/MapServer/2/query"
    ),

    # Confirmed via the LIDAR Tiles layer's extent (wkid 2038, latestWkid
    # 2961) and a real tile's LIDAR_Projects.PROJECTION attribute
    # ("NAD_1983_CSRS_UTM_Zone_20N", CGVD2013).
    "ns_lidar_epsg": 2961,

    # Confirmed live: returned 12 real images with lat/lon/heading for a
    # Halifax test address using this exact base URL + path + params.
    "kartaview_base_url": "https://api.openstreetcam.org",
    "kartaview_nearby_path": "/2.0/photo/",

    "image_search_radius_m": 75,
    "image_bearing_tolerance_deg": 35,

    # Confirmed against `roofer --help-all` (roofer 1.x) + a synthetic test run:
    # usage is `roofer [options] <pointcloud>... <polygon-source> <output-directory>`,
    # LoD2.2 is emitted by default, and output is a per-tile CityJSONSequence
    # file named after the tile's lower-left corner (e.g. 400000_4900000.city.jsonl),
    # not a fixed filename. Re-check if you upgrade roofer.
    "roofer_bin": "roofer",

    # Directories to search for already-downloaded GeoNOVA LAZ tiles (matched
    # by PRODNAME, e.g. "462_4949_201901.laz") before giving up -- GeoNOVA's
    # self-serve download isn't automated yet, so tiles fetched by hand from
    # https://nsgi.novascotia.ca/datalocator/elevation/ get picked up from here.
    "lidar_local_search_dirs": [str(Path.home() / "Downloads")],
}

WGS84 = "EPSG:4326"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Compass bearing (0=N, 90=E) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff_deg(a, b) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def external_tool_env() -> dict:
    """Environment for pdal/roofer subprocess calls.

    Both link PROJ for CRS handling. On this machine a stale system-wide
    PROJ_LIB (pointing at a 2015-era PROJ4 data folder with no proj.db --
    left behind by an unrelated GIS tool) made roofer fail outright with
    "PROJ: proj_create_from_database: Cannot find proj.db", even though a
    working proj.db existed elsewhere. So an existing PROJ_DATA/PROJ_LIB is
    validated (proj.db must actually be present in it), not just trusted
    because it's set -- confirmed by reproducing the failure and the fix
    against a synthetic test building.
    """
    env = os.environ.copy()
    existing = env.get("PROJ_DATA") or env.get("PROJ_LIB")
    if existing and (Path(existing) / "proj.db").exists():
        return env

    conda_prefix = env.get("CONDA_PREFIX")
    candidates = []
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "share" / "proj")
    candidates.append(Path.home() / "AppData" / "Local" / "miniconda3" / "Library" / "share" / "proj")
    for c in candidates:
        if (c / "proj.db").exists():
            env["PROJ_DATA"] = str(c)
            env["PROJ_LIB"] = str(c)
            break
    return env


# ---------------------------------------------------------------------------
# Stage 1: geocode
# ---------------------------------------------------------------------------

@dataclass
class GeocodeResult:
    lat: float
    lon: float
    display_name: str


def geocode(address: str) -> GeocodeResult:
    resp = requests.get(
        CONFIG["nominatim_url"],
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise RuntimeError(f"geocode: no match for {address!r}")
    r = results[0]
    return GeocodeResult(lat=float(r["lat"]), lon=float(r["lon"]), display_name=r["display_name"])


# ---------------------------------------------------------------------------
# Stage 2: footprint
# ---------------------------------------------------------------------------

def fetch_nearby_footprints(lat: float, lon: float, buffer_deg: float = 0.0006) -> list[dict]:
    """Query all HRM building footprints intersecting a small bbox around a
    point -- unlike fetch_footprint(), which returns only the single nearest
    one, this is for occlusion checks that need every building in an area.
    """
    envelope = {
        "xmin": lon - buffer_deg,
        "ymin": lat - buffer_deg,
        "xmax": lon + buffer_deg,
        "ymax": lat + buffer_deg,
        "spatialReference": {"wkid": 4326},
    }
    resp = requests.get(
        CONFIG["hrm_footprint_layer_url"],
        params={
            "geometry": json.dumps(envelope),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("features", [])


def fetch_footprint(lat: float, lon: float, buffer_deg: float = 0.0006) -> dict:
    """Query the HRM buildings layer for the polygon nearest (lat, lon).

    buffer_deg is a small WGS84 bbox pad (~50-70m at this latitude) around the
    point -- cheap way to catch the containing footprint without needing a
    point-in-polygon-capable query on the server side.
    """
    features = fetch_nearby_footprints(lat, lon, buffer_deg)
    if not features:
        raise RuntimeError("fetch_footprint: no building footprint found near point")

    point = shape({"type": "Point", "coordinates": [lon, lat]})
    best = min(features, key=lambda f: shape(f["geometry"]).distance(point))
    return best


def footprint_polygon(footprint_geojson: dict) -> Polygon:
    geom = shape(footprint_geojson["geometry"])
    return geom if geom.geom_type == "Polygon" else list(geom.geoms)[0]


# ---------------------------------------------------------------------------
# Stage 3: lidar crop
# ---------------------------------------------------------------------------

def find_covering_tiles(footprint_wgs84: Polygon) -> list[dict]:
    """Query GeoNOVA's LIDAR Tiles layer for tiles intersecting the footprint.

    Handles the case where a footprint straddles a tile boundary by returning
    every intersecting tile -- callers must merge/crop across all of them.
    Each result's attributes carry TILENUMB (e.g. "438_4946") and PRODNAME
    (e.g. "438_4946_201901.laz") confirmed against a live query, but there is
    no field with a ready-to-download URL -- see the TODO(verify) note on
    CONFIG["geonova_lidar_tiles_url"].
    """
    minx, miny, maxx, maxy = footprint_wgs84.bounds
    envelope = {
        "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
        "spatialReference": {"wkid": 4326},
    }
    resp = requests.get(
        CONFIG["geonova_lidar_tiles_url"],
        params={
            "geometry": json.dumps(envelope),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "geojson",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    fc = resp.json()
    tiles = fc.get("features", [])
    if not tiles:
        raise RuntimeError("find_covering_tiles: no lidar tile covers this footprint")
    return tiles


def reproject_footprint(footprint_wgs84: Polygon, dst_epsg: int) -> Polygon:
    """Reproject WGS84 footprint into the lidar tile CRS.

    Must happen before the PDAL crop -- cropping a UTM point cloud with a
    lon/lat polygon silently returns zero points (no error raised), which is
    the single most likely "empty building" failure mode in this pipeline.
    """
    transformer = Transformer.from_crs(WGS84, f"EPSG:{dst_epsg}", always_xy=True)
    coords = [transformer.transform(x, y) for x, y in footprint_wgs84.exterior.coords]
    return Polygon(coords)


def download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out_path


def dedupe_latest_tiles(tiles: list[dict]) -> list[dict]:
    """GeoNOVA's LIDAR Tiles layer returns one row per (tile, project) pair --
    the same TILENUMB shows up once per lidar vintage (e.g. 2007 and 2019).
    Keep only the newest project per tile number.
    """
    best_by_tile: dict[str, dict] = {}
    for tile in tiles:
        props = tile["properties"]
        tile_number = props.get("LIDAR_Tiles.TILENUMB")
        year = props.get("LIDAR_Projects.YEARDATE") or "0"
        current = best_by_tile.get(tile_number)
        if current is None or year > current["properties"].get("LIDAR_Projects.YEARDATE", "0"):
            best_by_tile[tile_number] = tile
    return list(best_by_tile.values())


def locate_local_lidar_file(prodname: str) -> Optional[Path]:
    for d in CONFIG["lidar_local_search_dirs"]:
        candidate = Path(d) / prodname
        if candidate.exists():
            return candidate
    return None


def crop_lidar_to_footprint(footprint_geojson: dict, work_dir: Path) -> Path:
    footprint_wgs84 = footprint_polygon(footprint_geojson)
    epsg = CONFIG["ns_lidar_epsg"]
    footprint_proj = reproject_footprint(footprint_wgs84, epsg)

    tiles = dedupe_latest_tiles(find_covering_tiles(footprint_wgs84))
    laz_paths = []
    for i, tile in enumerate(tiles):
        props = tile["properties"]
        tile_number = props.get("LIDAR_Tiles.TILENUMB")
        prodname = props.get("LIDAR_Tiles_Products.PRODNAME")

        cached = work_dir / "tiles" / (prodname or f"tile_{i}.laz")
        if cached.exists():
            laz_paths.append(cached)
            continue

        local_file = locate_local_lidar_file(prodname) if prodname else None
        if local_file is not None:
            laz_paths.append(local_file)
            continue

        # TODO(verify): GeoNOVA's LIDAR Tiles layer identifies which tiles
        # cover the footprint (confirmed live) but exposes no per-tile
        # download URL -- the DataLocator Elevation Explorer's self-serve
        # download is not a plain REST call found via search/API probing.
        # Either wire up its real download flow (inspect the browser network
        # tab while using https://nsgi.novascotia.ca/datalocator/elevation/)
        # or pre-fetch tiles manually into one of CONFIG["lidar_local_search_dirs"].
        raise RuntimeError(
            f"crop_lidar_to_footprint: no download mechanism wired up for tile "
            f"{tile_number!r} ({prodname!r}) and it wasn't found in "
            f"{CONFIG['lidar_local_search_dirs']}. Fetch it manually from "
            f"https://nsgi.novascotia.ca/datalocator/elevation/, or implement "
            f"the self-serve download flow."
        )

    wkt = footprint_proj.buffer(2.0).wkt  # small buffer so wall-adjacent points aren't clipped off
    out_laz = work_dir / "building_crop.laz"

    pipeline = {
        "pipeline": (
            [{"type": "readers.las", "filename": str(p)} for p in laz_paths]
            + ([{"type": "filters.merge"}] if len(laz_paths) > 1 else [])
            + [
                {
                    "type": "filters.crop",
                    "polygon": wkt,
                    "a_srs": f"EPSG:{epsg}",
                },
                # GeoNOVA's NS lidar is NOT pre-classified with a building code
                # (confirmed against a real tile: only ground/vegetation/noise
                # classes are present, no class 6) -- unlike GeoNB's NB data,
                # which the plan notes ships pre-classified. roofer selects
                # building points by --bld-class (default 6), so without this
                # reclassification it finds zero building points and silently
                # reconstructs "no points" roofs. Drop noise, then treat every
                # non-ground point inside the tight footprint buffer as building.
                {"type": "filters.range", "limits": "Classification![7:7]"},
                {"type": "filters.assign", "value": ["Classification = 6 WHERE Classification != 2"]},
                {"type": "writers.las", "filename": str(out_laz)},
            ]
        )
    }
    pipeline_path = work_dir / "crop_pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline, indent=2))

    subprocess.run(["pdal", "pipeline", str(pipeline_path)], check=True, env=external_tool_env())
    return out_laz


# ---------------------------------------------------------------------------
# Stage 4: street-level images
# ---------------------------------------------------------------------------

@dataclass
class StreetImage:
    id: str
    url: str
    lat: float
    lon: float
    heading_deg: float
    distance_m: float = 0.0
    bearing_from_image_deg: float = 0.0
    facing: bool = False


def fetch_kartaview_images(lat: float, lon: float) -> list[StreetImage]:
    resp = requests.get(
        CONFIG["kartaview_base_url"] + CONFIG["kartaview_nearby_path"],
        params={
            "lat": lat,
            "lng": lon,
            "radius": CONFIG["image_search_radius_m"],
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    images = []
    for item in data.get("result", {}).get("data", []):
        # KartaView returns some photos with "heading"/"lat"/"lng": null
        # (present but not missing) -- confirmed against live responses,
        # where .get(k, default) silently returns None instead of the
        # default. An image with no real heading can't be bearing-filtered
        # honestly (defaulting it to 0 would fabricate a direction and could
        # produce false "facing" matches), so it's skipped rather than
        # guessed at.
        lat, lng, heading = item.get("lat"), item.get("lng"), item.get("heading")
        if lat is None or lng is None or heading is None:
            continue
        images.append(
            StreetImage(
                id=str(item["id"]),
                url=item.get("fileurlProc") or item.get("fileurl"),
                lat=float(lat),
                lon=float(lng),
                heading_deg=float(heading),
            )
        )
    return images


def sightline_is_clear(image: StreetImage, building_lat: float, building_lon: float,
                        target_footprint: dict) -> bool:
    """Reject an image whose camera-to-target sightline passes through a
    *different*, closer building first.

    Bearing/distance alone can't tell "aimed at the right coordinates" from
    "a closer building is actually in frame" -- confirmed as a real failure
    mode: a KartaView photo correctly flagged as facing 1254 Hollis St by
    bearing/distance actually showed the neighbouring storefront (New Asia)
    instead, sitting almost exactly on the same sightline at nearly the same
    distance from the camera.
    """
    target_id = target_footprint.get("properties", {}).get("BL_ID")
    dist_to_target = haversine_m(image.lat, image.lon, building_lat, building_lon)
    sightline = LineString([(image.lon, image.lat), (building_lon, building_lat)])

    nearby = fetch_nearby_footprints(image.lat, image.lon, buffer_deg=0.0006)
    for feat in nearby:
        if feat.get("properties", {}).get("BL_ID") == target_id:
            continue
        poly = footprint_polygon(feat)
        if not sightline.intersects(poly):
            continue
        inter = sightline.intersection(poly)
        coords = list(inter.coords) if inter.geom_type in ("LineString", "Point") else []
        for ix, iy in coords:
            if haversine_m(image.lat, image.lon, iy, ix) < dist_to_target - 3.0:  # a few metres of slack
                return False
    return True


def filter_facing_images(images: list[StreetImage], building_lat: float, building_lon: float,
                          target_footprint: Optional[dict] = None) -> list[StreetImage]:
    """Keep only images whose camera heading points roughly at the building
    and, if a footprint is given, whose sightline isn't blocked by a closer
    building first (see sightline_is_clear).
    """
    tol = CONFIG["image_bearing_tolerance_deg"]
    kept = []
    for img in images:
        img.distance_m = haversine_m(img.lat, img.lon, building_lat, building_lon)
        img.bearing_from_image_deg = bearing_deg(img.lat, img.lon, building_lat, building_lon)
        img.facing = angle_diff_deg(img.heading_deg, img.bearing_from_image_deg) <= tol
        if not (img.facing and img.distance_m <= CONFIG["image_search_radius_m"]):
            continue
        if target_footprint is not None and not sightline_is_clear(img, building_lat, building_lon, target_footprint):
            continue
        kept.append(img)
    return kept


# ---------------------------------------------------------------------------
# Stage 5: roofer -> LOD 2.2
# ---------------------------------------------------------------------------

def write_projected_footprint_geojson(footprint_geojson: dict, epsg: int, out_path: Path) -> Path:
    """Write the footprint reprojected into roofer's working CRS.

    roofer's --srs flag overrides the CRS for input sources rather than
    reprojecting a GeoJSON's implicit WGS84 lon/lat -- confirmed by running
    a synthetic test with a WGS84-shaped footprint and a projected point
    cloud under the same --srs, which places the footprint wrong. The file
    handed to roofer must carry raw coordinates already in that CRS, same as
    the point cloud.
    """
    poly_wgs84 = footprint_polygon(footprint_geojson)
    poly_proj = reproject_footprint(poly_wgs84, epsg)
    feature = {
        "type": "Feature",
        "properties": footprint_geojson.get("properties", {}),
        "geometry": mapping(poly_proj),
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    out_path.write_text(json.dumps(fc))
    return out_path


def _shift_boundary_indices(boundaries: list, offset: int) -> None:
    """Recursively add offset to every vertex index in a CityJSON boundaries
    array, in place. Nesting depth varies by geometry type (Solid vs
    MultiSurface), so this walks to whatever depth the leaves are at."""
    for i, b in enumerate(boundaries):
        if isinstance(b, list):
            _shift_boundary_indices(b, offset)
        else:
            boundaries[i] = b + offset


def merge_cityjsonseq(path: Path) -> dict:
    """Merge a roofer CityJSONSequence (JSON Lines: one metadata line, then
    one CityJSONFeature per building) into a single plain CityJSON document
    with real-world vertex coordinates, so the rest of the pipeline can treat
    LOD2.2 output as an ordinary CityJSON dict instead of a line-delimited
    sequence. Confirmed against actual roofer output, not the CityJSON spec
    alone -- feature vertices are local integers scaled/translated by the
    metadata line's `transform`.
    """
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    meta = json.loads(lines[0])
    scale = meta["transform"]["scale"]
    translate = meta["transform"]["translate"]

    merged = {
        "type": "CityJSON",
        "version": meta.get("version", "2.0"),
        "metadata": meta.get("metadata", {}),
        "CityObjects": {},
        "vertices": [],
    }
    for line in lines[1:]:
        feat = json.loads(line)
        if feat.get("type") != "CityJSONFeature":
            continue
        offset = len(merged["vertices"])
        for vx, vy, vz in feat["vertices"]:
            merged["vertices"].append([
                vx * scale[0] + translate[0],
                vy * scale[1] + translate[1],
                vz * scale[2] + translate[2],
            ])
        for obj_id, obj in feat["CityObjects"].items():
            obj_copy = json.loads(json.dumps(obj))
            for geom in obj_copy.get("geometry", []):
                _shift_boundary_indices(geom.get("boundaries", []), offset)
            merged["CityObjects"][obj_id] = obj_copy
    return merged


def run_roofer(footprint_path: Path, laz_path: Path, out_dir: Path) -> Path:
    """Run roofer and merge its CityJSONSequence output into a single plain
    CityJSON file. Argument order (pointcloud, polygon-source, output-dir as
    a directory not a file), --srs, and the tile-named *.city.jsonl output
    are all confirmed against `roofer --help-all` plus a real run against a
    synthetic test building -- not guessed.
    """
    roofer_out_dir = out_dir / "roofer_raw"
    roofer_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        CONFIG["roofer_bin"],
        "--srs", f"EPSG:{CONFIG['ns_lidar_epsg']}",
        str(laz_path),
        str(footprint_path),
        str(roofer_out_dir),
    ]
    subprocess.run(cmd, check=True, env=external_tool_env())

    seq_files = sorted(roofer_out_dir.glob("*.city.jsonl"))
    if not seq_files:
        raise RuntimeError("run_roofer: no *.city.jsonl produced -- check roofer output above")

    merged = merge_cityjsonseq(seq_files[0])
    out_path = out_dir / "lod22.city.json"
    out_path.write_text(json.dumps(merged))
    return out_path


# ---------------------------------------------------------------------------
# Stage 6: openings from images -> LOD 3.1
# ---------------------------------------------------------------------------

@dataclass
class WallFace:
    surface_index: int
    normal: np.ndarray
    plane_point: np.ndarray
    azimuth_deg: float
    along: np.ndarray        # horizontal unit vector running along the wall
    along_min: float         # extent of the wall's own vertices, projected onto
    along_max: float         # `along` and made relative to plane_point
    z_min: float             # real vertical extent of the wall (not a guess)
    z_max: float


def extract_wall_faces(lod22_cityjson: dict) -> list[WallFace]:
    """Pull wall surfaces (semantics.type == 'WallSurface') out of the LOD2.2
    CityJSON and compute each one's plane (point + normal) and compass azimuth.
    """
    walls: list[WallFace] = []
    objects = lod22_cityjson.get("CityObjects", {})
    verts = np.array(lod22_cityjson.get("vertices", []), dtype=float)

    # CityJSON boundary/semantics nesting varies by geometry type (Solid vs
    # MultiSurface); walk it generically instead of assuming a fixed depth.
    # A surface's boundary is always a list of rings (b[0] = exterior ring,
    # any further entries are holes) -- confirmed against real roofer output.
    def walk(boundaries, sem_values, surfaces):
        if sem_values is None:
            return
        if isinstance(sem_values, int):
            return
        for b, s in zip(boundaries, sem_values):
            if isinstance(s, list):
                walk(b, s, surfaces)
            else:
                if s is not None and surfaces[s].get("type") == "WallSurface":
                    ring = b[0]
                    pts = verts[ring]
                    if len(pts) >= 3:
                        v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
                        normal = np.cross(v1, v2)
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            normal = normal / norm
                            az = (math.degrees(math.atan2(normal[0], normal[1])) + 360) % 360
                            center = pts.mean(axis=0)
                            along = np.array([-normal[1], normal[0]])
                            s_vals = (pts[:, :2] - center[:2]) @ along
                            walls.append(WallFace(
                                len(walls), normal, center, az, along,
                                float(s_vals.min()), float(s_vals.max()),
                                float(pts[:, 2].min()), float(pts[:, 2].max()),
                            ))

    for obj in objects.values():
        for geom in obj.get("geometry", []):
            semantics = geom.get("semantics", {})
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [])
            walk(geom.get("boundaries", []), values, surfaces)

    return walls


def merge_coplanar_walls(walls: list[WallFace], azimuth_tol_deg: float = 5.0, offset_tol_m: float = 1.0) -> list[WallFace]:
    """Merge wall fragments that share nearly the same azimuth and lie on
    nearly the same plane into single combined walls spanning their
    combined along-extent.

    Real facades often get split into several small fragments during LOD2.2
    reconstruction. Confirmed against 1259 Barrington St: its largest single
    wall fragment was only 5.8m, but groups of same-azimuth fragments
    (checked for near-zero perpendicular offset, so this doesn't merge
    genuinely separate parallel walls like a building's front and back)
    combine into much larger real walls -- e.g. six ~160.4 deg fragments sum
    to ~15.7m. Investigated whether this fragmentation was caused by the
    known vegetation-reclassification issue and found it likely isn't: the
    original NS classification's "high vegetation" points span the building's
    full height up to its actual roof ridge, meaning that classifier lumps
    real roof points in with vegetation rather than the reverse. Merging
    fixes the symptom (too-small walls to place openings on confidently)
    regardless of the fragmentation's root cause.
    """
    used = [False] * len(walls)
    merged: list[WallFace] = []
    for i, w in enumerate(walls):
        if used[i]:
            continue
        group = [w]
        used[i] = True
        for j in range(i + 1, len(walls)):
            if used[j]:
                continue
            w2 = walls[j]
            if angle_diff_deg(w.azimuth_deg, w2.azimuth_deg) > azimuth_tol_deg:
                continue
            offset = abs(np.dot(w2.plane_point[:2] - w.plane_point[:2], w.normal[:2]))
            if offset > offset_tol_m:
                continue
            group.append(w2)
            used[j] = True

        if len(group) == 1:
            merged.append(w)
            continue

        base = group[0]
        z_min = min(g.z_min for g in group)
        z_max = max(g.z_max for g in group)
        all_s = []
        for g in group:
            offset_along = np.dot(g.plane_point[:2] - base.plane_point[:2], base.along)
            all_s.append(offset_along + g.along_min)
            all_s.append(offset_along + g.along_max)
        merged.append(WallFace(
            base.surface_index, base.normal, base.plane_point, base.azimuth_deg,
            base.along, min(all_s), max(all_s), z_min, z_max,
        ))
    return merged


_grounding_dino = None  # lazy-loaded (processor, model, device) tuple
_depth_model = None     # lazy-loaded (processor, model, device) tuple


def _load_grounding_dino():
    global _grounding_dino
    if _grounding_dino is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        _grounding_dino = (processor, model, device)
    return _grounding_dino


def _load_depth_model():
    global _depth_model
    if _depth_model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Metric (not just relative) outdoor checkpoint -- gives depth in real
        # meters, so it can be compared directly against the camera-to-wall
        # distance already computed geometrically in backproject_opening.
        model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        _depth_model = (processor, model, device)
    return _depth_model


def estimate_depth_map(image: Image.Image) -> np.ndarray:
    """Per-pixel metric depth (metres) via Depth Anything V2's outdoor metric
    checkpoint.

    Needed because monocular back-projection has no depth signal of its own:
    confirmed against a real photo where Grounding DINO correctly found real
    windows on a distant, unrelated background building, and those
    detections' rays still landed within the *foreground* target wall's
    bounds (ray-plane intersection just solves for whatever depth hits that
    one assumed plane -- it can't know the photographed object is actually
    much farther away). This depth map is what lets detect_openings_in_image
    tag each box with an independent real-world distance estimate, so
    backproject_opening can reject the ones that don't match.
    """
    import torch
    processor, model, device = _load_depth_model()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]
    return depth.cpu().numpy()


def detect_openings_in_image(image_path: Path) -> list[dict]:
    """Zero-shot window/door detection via Grounding DINO, prompted with
    plain-text labels, plus per-class NMS to drop duplicate overlapping
    boxes. Replaces an earlier Canny/contour heuristic that had no semantic
    label at all -- confirmed against a real facade photo to correctly catch
    double-hung windows, storefront glass, and the actual entrance door.

    "balcony" is deliberately not a detection prompt here: tested against a
    real photo and found to produce large, confidently-wrong boxes on
    signage/awnings at any threshold loose enough to also catch real
    windows/doors. Balconies are instead recovered downstream by
    classify_opening_type()'s geometric check on the back-projected opening.

    Each returned rect also carries "estimated_depth_m", the median Depth
    Anything estimate within its box, used by backproject_opening to reject
    real-but-wrong-building detections (see estimate_depth_map).
    """
    import torch
    from torchvision.ops import nms

    processor, model, device = _load_grounding_dino()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text="window. door.", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=0.25, text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes, labels, scores = results["boxes"], results["text_labels"], results["scores"]
    keep_idx = []
    for cls in set(labels):
        idx = [i for i, l in enumerate(labels) if l == cls]
        cls_boxes = torch.stack([boxes[i] for i in idx])
        cls_scores = torch.stack([scores[i] for i in idx])
        keep = nms(cls_boxes, cls_scores, iou_threshold=0.4)
        keep_idx.extend(idx[k] for k in keep.tolist())

    depth_map = estimate_depth_map(image) if keep_idx else None

    img_w, img_h = image.size
    openings = []
    for i in keep_idx:
        x0, y0, x1, y1 = (float(v) for v in boxes[i])
        xi0, yi0, xi1, yi1 = int(x0), int(y0), max(int(x1), int(x0) + 1), max(int(y1), int(y0) + 1)
        openings.append({
            "px": x0, "py": y0, "pw": x1 - x0, "ph": y1 - y0,
            "img_w": img_w, "img_h": img_h,
            "confidence": float(scores[i]),
            "detected_label": labels[i],
            "estimated_depth_m": float(np.median(depth_map[yi0:yi1, xi0:xi1])),
        })
    return openings


DEPTH_MISMATCH_RATIO = 1.8  # reject a detection if estimated_depth_m and the geometric ray distance differ by more than this factor


def backproject_opening(rect: dict, image: StreetImage, wall: WallFace, hfov_deg: float = 80.0) -> Optional[list]:
    """Approximate back-projection of a pixel rectangle onto a wall plane.

    Uses a pinhole-camera azimuth model (camera heading +/- half-FOV maps
    linearly across image width) to get a ray bearing per pixel column, then
    intersects that ray (from the image's lat/lon, converted to the wall's
    local planar frame) with the wall's vertical plane. The image row maps
    onto the wall's own real z-extent (top of image -> top of wall, bottom ->
    base), and the horizontal hit point is clamped to the wall's own along-
    extent -- earlier versions used an arbitrary wall vertex plus a fixed 6m
    height guess, which let openings float above the roofline or off the
    side of the building entirely (visible in early renders). If the ray hits
    far outside the wall's own footprint, this is probably the wrong wall
    (crude nearest-azimuth matching) rather than an edge case worth clamping,
    so it's rejected instead of smeared onto the boundary.
    """
    # Use the *computed* bearing from camera GPS to the target building
    # (set by filter_facing_images) as the frame-center direction, not the
    # camera's self-reported compass heading. Confirmed as a real, sizeable
    # error source: one KartaView photo's heading was 20-25 degrees off from
    # the true GPS-computed bearing to 1259 Barrington St -- within the 35
    # degree "is this roughly facing" tolerance, but enough error at ~20m
    # distance to miss a 5-10m wall entirely. Phone compass readings in
    # crowdsourced data are known to be considerably less reliable than GPS
    # fixes, so the geometry (two real lat/lon points) is trusted over the
    # device's own compass.
    img_w = rect["img_w"]
    col_center = rect["px"] + rect["pw"] / 2
    frac = (col_center / img_w) - 0.5
    center_bearing = image.bearing_from_image_deg if image.bearing_from_image_deg else image.heading_deg
    ray_bearing = (center_bearing + frac * hfov_deg) % 360

    # Intersect the ray from the camera with the wall's plane (2D, in a local
    # ENU-ish frame centered on the wall's plane_point) using the plane normal.
    transformer = Transformer.from_crs(WGS84, f"EPSG:{CONFIG['ns_lidar_epsg']}", always_xy=True)
    cam_x, cam_y = transformer.transform(image.lon, image.lat)
    cam = np.array([cam_x, cam_y])
    ray_dir = np.array([math.sin(math.radians(ray_bearing)), math.cos(math.radians(ray_bearing))])

    plane_p = wall.plane_point[:2]
    plane_n = wall.normal[:2]
    denom = np.dot(ray_dir, plane_n)
    if abs(denom) < 1e-6:
        return None  # ray parallel to wall, can't intersect
    t = np.dot(plane_p - cam, plane_n) / denom
    if t <= 0:
        return None  # wall is behind the camera along this ray

    # Reject detections whose independently-estimated depth (Depth Anything,
    # see estimate_depth_map) doesn't roughly match this ray's geometric
    # distance to the wall. Confirmed necessary: a real photo had genuine
    # windows correctly detected on a distant background building, and their
    # rays still intersected the *foreground* target wall's plane just fine
    # (ray-plane math has no way to know the photographed object is actually
    # much farther away) -- depth is the only signal that catches this.
    estimated_depth_m = rect.get("estimated_depth_m")
    if estimated_depth_m is not None and estimated_depth_m > 0:
        ratio = estimated_depth_m / t
        if not (1 / DEPTH_MISMATCH_RATIO <= ratio <= DEPTH_MISMATCH_RATIO):
            return None

    hit_xy = cam + t * ray_dir

    wall_len = wall.along_max - wall.along_min
    s = np.dot(hit_xy - plane_p, wall.along)
    margin = wall_len * 0.5
    if s < wall.along_min - margin or s > wall.along_max + margin:
        return None  # hit point is nowhere near this wall -- likely the wrong wall
    s = min(max(s, wall.along_min), wall.along_max)

    row_center = rect["py"] + rect["ph"] / 2
    v_frac = row_center / rect["img_h"]
    wall_height = wall.z_max - wall.z_min
    z_center = wall.z_max - v_frac * wall_height

    half_w = min((rect["pw"] / rect["img_w"]) * 3.0, wall_len / 2)  # rough angular-width-to-meters scaling
    half_h = min((rect["ph"] / rect["img_h"]) * wall_height / 2, wall_height / 2)

    s1 = max(s - half_w, wall.along_min)
    s2 = min(s + half_w, wall.along_max)
    z1 = max(z_center - half_h, wall.z_min)
    z2 = min(z_center + half_h, wall.z_max)

    p1 = plane_p + s1 * wall.along
    p2 = plane_p + s2 * wall.along
    return [
        [p1[0], p1[1], z1],
        [p2[0], p2[1], z1],
        [p2[0], p2[1], z2],
        [p1[0], p1[1], z2],
    ]


def classify_opening_type(poly: list, wall: WallFace, detected_label: Optional[str] = None) -> str:
    """Classify a back-projected opening as a door, balcony, or window.

    Grounding DINO gives a real "window"/"door" label (see
    detect_openings_in_image) which is trusted first -- it's a genuine
    semantic detection, not a guess. Balconies are never a detection prompt
    (too noisy on signage/awnings), so they're always recovered here from
    real-world geometry: elevated off the wall's base and wider than tall
    (railing-shaped), overriding whatever label the detector gave. If no
    detected_label is available at all (e.g. a manually-built rect), falls
    back to the original geometric door heuristic.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    zs = [p[2] for p in poly]
    width_m = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    height_m = max(zs) - min(zs)
    height_above_base = min(zs) - wall.z_min

    if height_above_base > 0.5 and width_m >= 1.8 and width_m > height_m * 1.3:
        return "balcony"
    if detected_label in ("window", "door"):
        return detected_label
    if height_above_base <= 0.5 and height_m >= 1.6 and width_m <= 2.5 and height_m > width_m:
        return "door"
    return "window"


MIN_WALL_AREA_M2 = 4.0  # ignore sliver/fragment walls (over-segmentation noise) when picking a facing wall


def add_windows(lod22_path: Path, images: list[StreetImage], image_dir: Path, out_path: Path) -> Path:
    lod22 = json.loads(lod22_path.read_text())
    walls = merge_coplanar_walls(extract_wall_faces(lod22))

    lod31 = json.loads(json.dumps(lod22))  # deep copy, keep LOD2.2 untouched
    openings_added = []
    walls_with_coverage: set[int] = set()

    for image in images:
        img_path = image_dir / f"{image.id}.jpg"
        if not img_path.exists():
            try:
                download_file(image.url, img_path)
            except requests.RequestException:
                continue

        rects = detect_openings_in_image(img_path)
        if not rects or not walls:
            continue

        # naive nearest-facing-wall match by azimuth similarity, restricted to
        # walls big enough to be real facades -- without this, a heavily
        # over-segmented roof can match a tiny sliver wall whose azimuth
        # happens to line up, and every opening then gets rejected by
        # backproject_opening's own-footprint check since the ray never
        # actually lands on it. Matched using the GPS-computed bearing to the
        # building, not the camera's self-reported heading -- same
        # unreliable-compass reasoning as in backproject_opening.
        substantial = [w for w in walls if (w.along_max - w.along_min) * (w.z_max - w.z_min) >= MIN_WALL_AREA_M2]
        candidates = substantial or walls
        match_bearing = image.bearing_from_image_deg if image.bearing_from_image_deg else image.heading_deg
        wall = min(candidates, key=lambda w: angle_diff_deg(w.azimuth_deg, (match_bearing + 180) % 360))
        walls_with_coverage.add(wall.surface_index)

        for rect in rects:
            poly = backproject_opening(rect, image, wall)
            if poly is None:
                continue
            openings_added.append({
                "wall_surface_index": wall.surface_index,
                "type": classify_opening_type(poly, wall, rect.get("detected_label")),
                "polygon": poly,
                "confidence": rect["confidence"],
                "source_image_id": image.id,
                "approximate": True,
            })

    lod31["+lod31_openings"] = openings_added
    lod31["+wall_coverage"] = {
        str(w.surface_index): ("detected" if w.surface_index in walls_with_coverage else "no-coverage")
        for w in walls
    }
    lod31["+opening_type_counts"] = {
        t: sum(1 for o in openings_added if o["type"] == t) for t in ("window", "door", "balcony")
    }
    lod31.setdefault("metadata", {})["lod"] = "3.1"

    out_path.write_text(json.dumps(lod31, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Stage 7: manifest
# ---------------------------------------------------------------------------

def write_manifest(work_dir: Path, address: str, geo: GeocodeResult, footprint_geojson: dict,
                    laz_path: Path, lod22_path: Path, lod31_path: Path,
                    images: list[StreetImage]) -> Path:
    manifest = {
        "address": address,
        "geocode": {"lat": geo.lat, "lon": geo.lon, "display_name": geo.display_name},
        "footprint": mapping(footprint_polygon(footprint_geojson)),
        "point_cloud": str(laz_path.relative_to(work_dir)),
        "lod22": str(lod22_path.relative_to(work_dir)),
        "lod31": str(lod31_path.relative_to(work_dir)),
        "images": [
            {
                "id": img.id, "url": img.url, "lat": img.lat, "lon": img.lon,
                "heading_deg": img.heading_deg, "distance_m": round(img.distance_m, 1),
            }
            for img in images
        ],
        "notes": "LOD 3.1 openings are image-derived and approximate; walls without "
                 "facing imagery are tagged no-coverage, not windowless.",
    }
    out_path = work_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(address: str, out_dir: str = "output") -> Path:
    work_dir = Path(out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] geocoding {address!r}")
    geo = geocode(address)
    print(f"      -> {geo.lat:.6f}, {geo.lon:.6f} ({geo.display_name})")

    print("[2/7] fetching building footprint")
    footprint_geojson = fetch_footprint(geo.lat, geo.lon)

    print("[3/7] cropping lidar to footprint")
    laz_path = crop_lidar_to_footprint(footprint_geojson, work_dir)

    print("[4/7] fetching + filtering street-level images")
    all_images = fetch_kartaview_images(geo.lat, geo.lon)
    facing_images = filter_facing_images(all_images, geo.lat, geo.lon, footprint_geojson)
    print(f"      -> {len(facing_images)}/{len(all_images)} images face the building")

    print("[5/7] running roofer for LOD 2.2")
    footprint_path = write_projected_footprint_geojson(
        footprint_geojson, CONFIG["ns_lidar_epsg"], work_dir / "footprint.geojson"
    )
    lod22_path = run_roofer(footprint_path, laz_path, work_dir)

    print("[6/7] detecting + back-projecting openings -> LOD 3.1")
    lod31_path = add_windows(lod22_path, facing_images, work_dir / "images", work_dir / "lod31.city.json")

    print("[7/7] writing manifest")
    manifest_path = write_manifest(work_dir, address, geo, footprint_geojson, laz_path,
                                    lod22_path, lod31_path, facing_images)

    print(f"done -> {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python halifax_lod_demo.py \"<street address>\" [output_dir]")
        sys.exit(1)
    address_arg = sys.argv[1]
    out_dir_arg = sys.argv[2] if len(sys.argv) > 2 else "output"
    run_pipeline(address_arg, out_dir_arg)
