# Halifax / Moncton Building LOD 3.1 — Project Summary

A reference doc for the application: what we're building, the data sources, the
decisions, and what still needs wiring.

---

## 1. Goal

From a **street address** (Halifax first, Moncton second), automatically produce a
**LOD 3.1 building model** — where:

> **LOD 3.1 = LOD 2.2 (roof + massing shell) + façade openings (windows/doors)**

The tool is fully automatic: address in → footprint + lidar + street images fetched
from open APIs → LOD 2.2 reconstructed → openings detected from images and placed on
the walls → LOD 3.1 out. Openings are **approximate** (image-derived) by design, and
the output declares that.

---

## 2. Final pipeline

```
address
  → geocode (Nominatim)
  → footprint / roofprint            (HRM Open Data building footprints)
  → lidar point cloud                (GeoNOVA Elevation Explorer tile → crop to footprint)
  → street-level images              (KartaView, filtered to images facing the building)
  → roofer                           → LOD 2.2 CityJSON (roof + massing)
  → detect openings on façade images + back-project onto walls
  → LOD 3.1 CityJSON  (openings tagged `approximate` + confidence; walls tagged
                       detected / no-coverage)
  → manifest.json for the viewer
```

**Physics that shapes the whole design:** airborne lidar is 2.5D — it captures the
**roof and ground** well and **walls barely at all**. So lidar builds the shell; every
**opening comes from the images**, never from the point cloud. No point count fixes
this — it's the wrong surface, not too few points.

---

## 3. Data sources (all free / open)

| Data | Source | Format / CRS | License |
|---|---|---|---|
| Geocoding | Nominatim (OSM) | JSON | ODbL / free (rate-limit, set UA) |
| Building footprints | **HRM Open Data** (Halifax) | ArcGIS REST → GeoJSON | Open (municipal) |
| Footprints (national fallback) | NRCan **Automatically Extracted Buildings** | GPKG/SHP/GeoParquet | OGL-Canada |
| LiDAR point cloud (NS) | **GeoNOVA** Elevation Explorer | LAZ, 1 km² tiles, NAD83(CSRS)/UTM 20N, CGVD2013 | Unrestricted |
| LiDAR point cloud (national) | NRCan **CanElevation** | LAZ / COPC | OGL-Canada |
| LiDAR point cloud (NB / Moncton) | **GeoNB** | LAZ, **pre-classified** (building = class 6) | Open |
| Street-level images | **KartaView** (also Mapillary) | API | CC-BY-SA |
| Reconstruction tool | **roofer** (3DBAG / TU Delft) | CityJSON out; LoD1.2/1.3/**2.2** | **GPLv3** |

**Moncton bonus:** NB lidar is pre-classified, so building points can be isolated by
classification (class 6) — sometimes without even needing a footprint clip.

**Licensing notes to carry forward:**
- **OGL-Canada** (NRCan) and the provincial portals: commercial use OK **with attribution**.
- **CC-BY-SA** (KartaView/Mapillary): derivatives/commercial allowed, but **share-alike**
  — a model *derived from* the imagery may inherit the open license (may not stay
  proprietary). Flagged for legal review.
- **roofer GPLv3**: GPL covers the *software*, not the *data it outputs*. The models
  roofer produces are yours. Running it server-side as a tool (SaaS, not distributing the
  binary) generally does **not** make your app GPL. Confirm with counsel.

---

## 4. Ruled out (and why)

- **Google Photorealistic 3D / Earth / 3D Tiles / Street View** — Google's terms
  **explicitly prohibit** using their imagery/output to reconstruct 3D models. Using it —
  even "just the openings", even via user screenshots, even as a non-profit/research lab —
  is the single most clearly-forbidden use. Also technically poor (their 3D is already a
  photogrammetric mesh) and no rural coverage. **Dead end for model-building.**
- **Google Aerial View API** — US addresses only; unusable for Canada.
- **Screenshot-and-upload workaround** — doesn't cure the ToS problem; the restriction
  attaches to *use of the imagery*, not to who clicked. Non-profit/research status doesn't
  fix it because the deliverable is an *operational application*, which the noncommercial
  tiers (incl. Earth Engine) specifically exclude.
- **Commercial oblique providers** (Nearmap / EagleView / Vexcel) — **not ruled out**, but
  paid. They *do* provide true four-sided oblique views and *can* license derivative/model
  rights — you pay for the permission Google withholds. Custom quote only; scope a tight
  AOI and ask for standard-vs-derivative pricing side by side. Kept as the paid fallback.

---

## 5. Cost

- The chosen stack (open Canadian data + KartaView + roofer) is **$0**.
- If Google APIs were ever used for *reference viewing only*: pay-as-you-go, per-SKU free
  tiers (Essentials 10k/mo, Pro 5k/mo, Enterprise 3D Tiles 1k/mo). Always set a **Console
  quota cap** (hard stop) + budget alert — no default spend ceiling exists. Not needed for
  the current design.

---

## 6. Image → openings (the "3.1" step)

Approximate, image-only, automatic:

1. Keep façade images that **face the building** (bearing from image lat/lon to building
   vs. the image heading, within tolerance + distance) — pure metadata, no pixels.
2. **Detect** window/door rectangles per façade image (OpenCV placeholder now; swap in a
   trained window/door detector for real quality).
3. **Back-project** each pixel rectangle into 3D: intersect the camera ray (image pose =
   lat/lon + heading) with the matching **LOD 2.2 wall plane** to place the opening.
4. **Tag** each opening `approximate` + confidence; tag each wall `detected` vs
   `no-coverage` (KartaView is crowdsourced — some walls have no facing image; never render
   those as windowless without the flag).

Geometry proves the camera was *aimed* at the building, not that it's unoccluded — add a
vision/occlusion check later if needed.

---

## 7. Open TODOs (verify against live docs — do NOT fabricate)

- KartaView API base + nearby-photos endpoint/params (renamed from OpenStreetCam).
- GeoNOVA / CanElevation tile-index service URL + tile URL scheme; two-tile-span handling.
- HRM building-footprint ArcGIS REST layer URL.
- Exact `roofer` CLI flags (`roofer --help` for the installed build).
- Confirm NS tile EPSG (NAD83(CSRS)/UTM 20N — likely **EPSG:2961**).
- Replace OpenCV opening-detection stub with a trained detector when moving past demo.

**CRS gotcha:** the footprint must be reprojected into the tile CRS *before* the PDAL
crop, or the crop silently returns **zero points** (no error). First thing to check if a
building comes back empty.

---

## 8. Architecture note

Reconstruction is **server-side** — PDAL (crop) and roofer (model) are not browser tools.
Split: a **backend pipeline** (this scaffold) + a thin **viewer** that reads `manifest.json`
and displays footprint + point cloud + selected images + LOD 2.2/3.1 model. The viewer can
largely run in-browser (deck.gl / MapLibre); the pipeline cannot.

---

## 9. Scaffold

`halifax_lod_demo.py` — staged backend implementing all of the above. Correct as written:
stage structure, geocode, bearing/image-filter math, PDAL crop, pyproj reprojection.
`TODO(verify)` markers on the four endpoints above. `add_windows()` is the approximate
opening step. Runs locally with PDAL + roofer + network; not in a chat sandbox.

**Definition of done:** address → auto-fetch footprint + lidar + images → roofer LOD 2.2
→ image-detected openings back-projected onto walls → LOD 3.1 CityJSON, openings
`approximate` + confidence, walls flagged for coverage. Test on 3 real Halifax addresses:
footprint returned, point count > 0, ≥1 facing image where coverage exists, valid CityJSON,
≥1 wall with openings.