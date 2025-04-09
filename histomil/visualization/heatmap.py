import numpy as np
from openslide import OpenSlide


def compute_attention_map(
    attention_scores: np.array,
    tile_ids: list,
    tile_size: int = 224,
    tile_mpp: float = 1.0,
    wsi_path: str = None,
    output_mpp: float = 2.0,
    return_thumbnail: bool = True,
) -> np.array:
    attention_scores = np.squeeze(attention_scores)
    wsi = OpenSlide(wsi_path)
    mpp_x, mpp_y = (
        wsi.properties.get("openslide.mpp-x"),
        wsi.properties.get("openslide.mpp-y"),
    )
    if mpp_x is None or mpp_y is None:
        raise ValueError("Microns per pixel not found in WSI properties.")
    if mpp_x != mpp_y:
        raise ValueError("Microns per pixel values are not equal.")

    mpp_x = float(mpp_x)
    resizing_factor = mpp_x / output_mpp
    wsi_width, wsi_height = wsi.level_dimensions[0]
    width = int(wsi_width * resizing_factor)
    height = int(wsi_height * resizing_factor)

    attention_map = np.zeros((height, width), dtype=np.float32)
    resized_tile_size = int(tile_size * resizing_factor * tile_mpp / mpp_x)
    for tile_idx, tile_id in enumerate(tile_ids):
        x, y = get_position_from_tile_id(tile_id)
        resized_x = int(x * resizing_factor)
        resized_y = int(y * resizing_factor)
        attention_map[
            resized_y : resized_y + resized_tile_size,
            resized_x : resized_x + resized_tile_size,
        ] = attention_scores[tile_idx]

    if return_thumbnail:
        thumbnail = wsi.get_thumbnail((width, height))
        return attention_map, thumbnail
    return attention_map


def get_position_from_tile_id(tile_id):
    tile_id = tile_id.decode("utf-8")
    parts = tile_id.split("__x")[1].split("_y")
    x = int(parts[0])
    y = int(parts[1])
    return x, y

