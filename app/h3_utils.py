import h3
from .config import settings

def cell_for_point(lat: float, lon: float, resolution: int) -> str:
    return h3.latlng_to_cell(lat, lon, resolution)

def cell_geometry_wkt(cell: str) -> str:
    # H3 returnează (lat, lon); WKT folosește (lon lat).
    boundary = h3.cell_to_boundary(cell)
    coords = [(lon, lat) for lat, lon in boundary]
    coords.append(coords[0])
    return "POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in coords) + "))"

def cell_center_wkt(cell: str) -> str:
    lat, lon = h3.cell_to_latlng(cell)
    return f"POINT({lon} {lat})"

def station_cells(lat: float, lon: float) -> tuple[str, str]:
    return (
        cell_for_point(lat, lon, settings.h3_res8),
        cell_for_point(lat, lon, settings.h3_res9),
    )
