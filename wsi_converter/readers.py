import multiprocessing
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from wsic.codecs import register_codecs
from wsic.enums import Codec, ColorSpace
from wsic.magic import summon_file_types
from wsic.metadata import ngff
from wsic.typedefs import PathLike
from wsic.utils import (
    TimeoutWarning,
    block_downsample_shape,
    main_process,
    mean_pool,
    mosaic_shape,
    ppu2mpp,
    resize_array,
    scale_to_fit,
    tile_slices,
)


class Reader(ABC):
    """Base class for readers."""

    def __init__(self, path: PathLike):
        """Initialize reader.

        Args:
            path (PathLike):
                Path to file.
        """
        self.path = path

    @abstractmethod
    def __getitem__(self, index: Tuple[Union[int, slice], ...]) -> np.ndarray:
        """Get pixel data at index."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_file(cls, path: Path) -> "Reader":
        """Return reader for file.

        Args:
            path (Path): Path to file.

        Returns:
            Reader: Reader for file.
        """
        path = Path(path)
        file_types = summon_file_types(path)
        raise ValueError(f"Unsupported file type: {path}")

    def thumbnail(self, shape: Tuple[int, ...], approx_ok: bool = False) -> np.ndarray:
        """Generate a thumbnail image of (or near) the requested shape.

        Args:
            shape (Tuple[int, ...]):
                Shape of the thumbnail.
            approx_ok (bool):
                If True, return a thumbnail that is approximately the
                requested shape. It will be equal to or larger than the
                requested shape (the next largest shape possible via
                an integer block downsampling).

        Returns:
            np.ndarray: Thumbnail.
        """
        # NOTE: Assuming first two are Y and X
        yx_shape = self.shape[:2]
        yx_tile_shape = (
            self.tile_shape[:2] if self.tile_shape else np.minimum(yx_shape, (256, 256))
        )
        self_mosaic_shape = self.mosaic_shape or mosaic_shape(yx_shape, yx_tile_shape)
        (
            downsample_shape,
            downsample_tile_shape,
            downsample,
        ) = self._find_thumbnail_downsample(shape, yx_shape, yx_tile_shape)
        # NOTE: Assuming channels last
        channels = self.shape[-1]
        thumbnail = np.zeros(downsample_shape + (channels,), dtype=np.uint8)
        # Resize tiles to new_downsample_tile_shape and combine
        tile_indexes = list(np.ndindex(self_mosaic_shape))
        for tile_index in self.pbar(tile_indexes, desc="Generating thumbnail"):
            try:
                tile = self.get_tile(tile_index)
            except (ValueError, NotImplementedError):  # e.g. Not tiled
                tile = self[tile_slices(tile_index, yx_tile_shape)]
            tile = mean_pool(tile.astype(float), downsample).astype(np.uint8)
            # Make sure the tile being written will not exceed the
            # bounds of the thumbnail
            yx_position = tuple(
                i * size for i, size in zip(tile_index, downsample_tile_shape)
            )
            max_y, max_x = (
                min(tile_max, thumb_max - position)
                for tile_max, thumb_max, position in zip(
                    tile.shape, thumbnail.shape, yx_position
                )
            )
            sub_tile = tile[:max_y, :max_x]
            thumbnail[tile_slices(tile_index, downsample_tile_shape)] = sub_tile
        return thumbnail if approx_ok else resize_array(thumbnail, shape, "bicubic")

    @staticmethod
    def _find_thumbnail_downsample(
        thumbnail_shape: Tuple[int, int],
        yx_shape: Tuple[int, int],
        yx_tile_shape: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        """Find the downsample and tile shape for a thumbnail.

        Args:
            thumbnail_shape (Tuple[int, int]):
                Shape of the thumbnail to be generated.
            yx_shape (Tuple[int, int]):
                Shape of the image in Y and X.
            yx_tile_shape (Tuple[int, int]):
                Shape of the tiles in Y and X which will be used to
                generate the thumbnail.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], int]:
                Shape of the downsampled image, shape of the downsampled
                tiles, and the downsample factor.
        """
        downsample_shape = yx_shape
        downsample_tile_shape = yx_tile_shape
        downsample = 0
        while True:
            new_downsample = downsample + 1
            next_downsample_shape, new_downsample_tile_shape = block_downsample_shape(
                yx_shape, new_downsample, yx_tile_shape
            )
            if all(x <= max(0, y) for x, y in zip(downsample_shape, thumbnail_shape)):
                break
            downsample_shape = next_downsample_shape
            downsample_tile_shape = new_downsample_tile_shape
            downsample = new_downsample
        return downsample_shape, downsample_tile_shape, downsample

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        # Naive base implementation using __getitem__
        if not decode:
            raise NotImplementedError(
                "Fetching tiles without decoding is not supported."
            )
        if not hasattr(self, "tile_shape"):
            raise ValueError(
                "Cannot get tile from a non-tiled reader"
                " (must have attr 'tile_shape')."
            )
        slices = tile_slices(index, self.tile_shape)
        return self[slices]

    @staticmethod
    def pbar(iterable: Iterable, *args, **kwargs) -> Iterator:
        """Return an iterator that displays a progress bar.

        Uses tqdm if installed, otherwise falls back to a simple iterator.

        Args:
            iterable (Iterable):
                Iterable to iterate over.
            args (tuple):
                Positional arguments to pass to tqdm.
            kwargs (dict):
                Keyword arguments to pass to tqdm.

        Returns:
            Iterator: Iterator that displays a progress bar.
        """
        try:
            from tqdm.auto import tqdm
        except ImportError:

            def tqdm(x, *args, **kwargs):
                return x

        return tqdm(iterable, *args, **kwargs)

    @property
    def original_shape(self) -> Tuple[int, ...]:
        """Return the original shape of the image."""
        return self.shape


class OpenSlideReader(Reader):
    """Reader for OpenSlide files using openslide-python."""

    def __init__(self, path: Path) -> None:
        import openslide

        super().__init__(path)
        self.os_slide = openslide.OpenSlide(str(path))
        self.shape = self.os_slide.level_dimensions[0][::-1] + (3,)
        self.dtype = np.uint8
        self.axes = "YXS"
        self.tile_shape = None  # No easy way to get tile shape currently
        self.microns_per_pixel = self._get_mpp()
        self.mosaic_shape = None

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        raise NotImplementedError("OpenSlideReader does not support reading tiles.")

    def _get_mpp(self) -> Optional[Tuple[float, float]]:
        """Get the microns per pixel for the image.

        Returns:
            Optional[Tuple[float, float]]:
                The microns per pixel as (x, y) tuple.
        """
        try:
            return (
                float(self.os_slide.properties["openslide.mpp-x"]),
                float(self.os_slide.properties["openslide.mpp-y"]),
            )
        except KeyError:
            warnings.warn("OpenSlide could not find MPP.", stacklevel=2)
        # Fall back to TIFF resolution tags
        try:
            resolution = (
                float(self.os_slide.properties["tiff.XResolution"]),
                float(self.os_slide.properties["tiff.YResolution"]),
            )
            units = self.os_slide.properties["tiff.ResolutionUnit"]
            self._check_sensible_resolution(resolution, units)
            return tuple(ppu2mpp(x, units) for x in resolution)
        except KeyError:
            warnings.warn("No resolution metadata found.", stacklevel=2)
        return None

    @staticmethod
    def _check_sensible_resolution(
        tiff_resolution: Tuple[float, float], tiff_units: int
    ) -> None:
        """Check whether the resolution is sensible.

        It is common for TIFF files to have incorrect resolution tags.
        This method checks whether the resolution is sensible and warns
        if it is not.

        Args:
            tiff_resolution (Tuple[float, float]):
                The TIFF resolution as an (x, y) tuple.
            tiff_units (int):
                The TIFF units of the resolution. A value of 2 indicates
                inches and a value of 3 indicates centimeters.
        """
        if tiff_units == 2 and 72 in tiff_resolution:
            warnings.warn(
                "TIFF resolution tags found."
                " However, they have a common default value of 72 pixels per inch."
                " This may from a misconfigured software library or tool"
                " which is expecting to handle print documents.",
                stacklevel=2,
            )
        if 0 in tiff_resolution:
            warnings.warn(
                "TIFF resolution tags found."
                " However, one or more of the values is zero.",
                stacklevel=2,
            )

    def __getitem__(self, index: Tuple[Union[int, slice], ...]) -> np.ndarray:
        """Get pixel data at index."""
        if index is ...:
            return np.array(self.os_slide.get_thumbnail(self.os_slide.dimensions))
        xs = index[1]
        ys = index[0]
        start_x = xs.start or 0
        start_y = ys.start or 0
        end_x = xs.stop or self.shape[1]
        end_y = ys.stop or self.shape[0]

        # Prevent reading past the edges of the image
        end_x = min(end_x, self.shape[1])
        end_y = min(end_y, self.shape[0])

        # Read the image
        img = self.os_slide.read_region(
            location=(start_x, start_y),
            level=0,
            size=(end_x - start_x, end_y - start_y),
        )
        return np.array(img.convert("RGB"))