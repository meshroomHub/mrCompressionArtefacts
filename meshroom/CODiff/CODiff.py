__version__ = "2.0"

from functools import total_ordering
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

from typing import Tuple

class CODiffBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class CODiff(desc.Node):
    category = "Compression Artefacts"
    documentation = """This node implements the CODiff vision tools."""
    
    gpu = desc.Level.EXTREME

    size = avpar.DynamicViewsSize("inputImages")
    parallelization = CODiffBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to process in a sfmData filepath",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input.",
            values=["jpg", "jpeg", "png", "exr"],
            value="jpg",
            exclusive=True,
        ),
        desc.IntParam(
            name="vaeEncoderTileSize",
            label="VAE Encoder Tile Size",
            value=1024,
            description="Tile size for VAE encoding",
            range=(64, 2048, 8),
            advanced=True,
        ),
        desc.IntParam(
            name="vaeDecoderTileSize",
            label="VAE Decoder Tile Size",
            value=224,
            description="Tile size for VAE decoding",
            range=(14, 448, 2),
            advanced=True,
        ),
        desc.IntParam(
            name="latentTileSize",
            label="Latent Tile Size",
            value=96,
            description="Tile size of latent space",
            range=(6, 192, 2),
            advanced=True,
        ),
        desc.IntParam(
            name="latentOverlapSize",
            label="Latent Overlap Size",
            value=32,
            description="Tile overlap in latent space",
            range=(2, 64, 2),
            advanced=True,
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the HDR images saved as hdr files.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="enhancedImage",
            label="Enhanced Image",
            description="Enhanced image.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>.exr",
        ),
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):
        import utils.utils_image as utils
        from diffusion.codiff import CODiff_test
        from diffusion.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

        from PIL import Image
        import OpenImageIO as oiio

        import torch
        import torch.nn.functional as F
        #from torch.utils import data
        from torchvision import transforms
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path
        import argparse

        def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
            if not torch.is_tensor(image):
                raise ValueError(f"Invalid input type={type(image)}.")
            if not torch.is_floating_point(image):
                raise ValueError(f"Invalid input dtype={image.dtype}.")
            if image.dim() != 4:
                raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

            h, w = image.shape[-2:]
            ph, pw = -h % align, -w % align

            image = F.pad(image, (0, pw, 0, ph), mode="replicate")

            return image, (ph, pw)

        def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
            if not torch.is_tensor(image):
                raise ValueError(f"Invalid input type={type(image)}.")
            if not torch.is_floating_point(image):
                raise ValueError(f"Invalid input dtype={image.dtype}.")
            if image.dim() != 4:
                raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

            ph, pw = padding
            uh = None if ph == 0 else -ph
            uw = None if pw == 0 else -pw

            image = image[:, :, :uh, :uw]

            return image


        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            chunk.logger.info("Loading CODiff model...")

            pretrained_model = os.getenv("STABLEDIFFUSION_WEIGHTS_PATH")
            codiff_path = os.getenv("CODIFF_MODELS_PATH") + "/codiff.pkl"
            cave_path = os.getenv("CODIFF_MODELS_PATH") + "/cave.pth"

            parser_codiff = argparse.ArgumentParser()
            parser_codiff.add_argument('--pretrained_model', type=str, default=pretrained_model)
            parser_codiff.add_argument("--codiff_path", type=str, default=codiff_path)
            # precision setting
            parser_codiff.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16") # 'fp32' KO !!!
            # merge lora
            parser_codiff.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
            # tile setting
            # parser_codiff.add_argument("--vae_decoder_tiled_size", type=int, default=224)
            # parser_codiff.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
            # parser_codiff.add_argument("--latent_tiled_size", type=int, default=96)
            # parser_codiff.add_argument("--latent_tiled_overlap", type=int, default=32)
            parser_codiff.add_argument("--vae_decoder_tiled_size", type=int, default=chunk.node.vaeDecoderTileSize.value)
            parser_codiff.add_argument("--vae_encoder_tiled_size", type=int, default=chunk.node.vaeEncoderTileSize.value)
            parser_codiff.add_argument("--latent_tiled_size", type=int, default=chunk.node.latentTileSize.value)
            parser_codiff.add_argument("--latent_tiled_overlap", type=int, default=chunk.node.latentOverlapSize.value)

            args_codiff = parser_codiff.parse_args([])

            # initialize the model
            model = CODiff_test(args_codiff)

            from cave.cave import CaVE

            cave = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR')
            cave.load_state_dict(torch.load(cave_path), strict=True)
            cave.eval()
            for k, v in cave.named_parameters():
                v.requires_grad = False
            cave = cave.to("cuda")

            # # weight type
            # weight_dtype = torch.float32
            # if args_codiff.mixed_precision == "fp16":
            #     weight_dtype = torch.float16


            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrEnhance:DeepModelName"] = "CODiff"
            metadata_deep_model["Meshroom:mrEnhance:DeepModelVersion"] = "0.1"

            for idx, path in enumerate(chunk_image_paths):
                with torch.no_grad():
                    image1, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), True)

                    h,w,c = image1.shape

                    w_tgt = h - h % 8
                    w_tgt = w - w % 8

                    oiio_image_buf = oiio.ImageBuf(image1)
                    oiio_image_buf = oiio.ImageBufAlgo.resize(oiio_image_buf, roi=oiio.ROI(0, w_tgt, 0, w_tgt, 0, 1, 0, c+1))
                    image1 = oiio_image_buf.get_pixels(format=oiio.FLOAT)

                    lq_raw = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to("cuda")     # 0, 1
                    lq = lq_raw * 2 - 1     # -1, 1

                    # lq_raw_padded, pad = pad_image(lq_raw, 8)
                    # lq_raw_padded = lq_raw_padded.to("cuda")     # 0, 1
                    # lq = lq_raw_padded * 2 - 1     # -1, 1

                    visual_embedding = cave.get_visual_embedding(lq_raw)
                    img_E = model(lq, visual_embedding)

                    # img_E_padded = model(lq, visual_embedding)
                    # img_E = unpad_image(img_E_padded, pad)

                    img_E = transforms.ToPILImage()(img_E[0].cpu() * 0.5 + 0.5)
                    #if args.align_method == 'adain':
                    img_E = adain_color_fix(target=img_E, source=image1)
                    #elif args.align_method == 'wavelet':
                    #    img_E = wavelet_color_fix(target=img_E, source=img_L)
                    #else:
                    #    pass

                    img_E = np.array(img_E) / 255.0

                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem

                    of_file_name = image_stem + ".exr"

                    image.writeImage(str(outputDirPath / of_file_name), img_E.copy(), h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)

            chunk.logger.info('CODiff end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path.")
    return image_paths
