import io
from dataclasses import dataclass

import numpy as np
import onnxruntime
from PIL import Image


@dataclass
class CloudInfo:
    name: str
    short: str
    url: str


class CloudNetInfer:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.setup()

        # Name labels by index
        self.labels = [
            "Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"
        ]

        self.labels_info = [
            CloudInfo(
                name="Altocumulus",
                short="Ac",
                url="https://cloudatlas.wmo.int/en/altocumulus-ac.html",
            ),
            CloudInfo(
                name="Altostratus",
                short="As",
                url="https://cloudatlas.wmo.int/en/altostratus-as.html",
            ),
            CloudInfo(
                name="Cumulonimbus",
                short="Cb",
                url="https://cloudatlas.wmo.int/en/cumulonimbus-cb.html",
            ),
            CloudInfo(
                name="Cirrocumulus",
                short="Cc",
                url="https://cloudatlas.wmo.int/en/cirrocumulus-cc.html",
            ),
            CloudInfo(
                name="Cirrus",
                short="Ci",
                url="https://cloudatlas.wmo.int/en/cirrus-ci.html",
            ),
            CloudInfo(
                name="Cirrostratus",
                short="Cs",
                url="https://cloudatlas.wmo.int/en/cirrostratus-cs.html",
            ),
            CloudInfo(
                name="Contrail",  # don't cloud
                short="Ct",
                url="https://cloudatlas.wmo.int/en/aircraft-condensation-trails.html",  # noqa: E501
            ),
            CloudInfo(
                name="Cumulus",
                short="Cu",
                url="https://cloudatlas.wmo.int/en/cumulus-cu.html",
            ),
            CloudInfo(
                name="Nimbostratus",
                short="Ns",
                url="https://cloudatlas.wmo.int/en/nimbostratus-ns.html",
            ),
            CloudInfo(
                name="Stratocumulus",
                short="Sc",
                url="https://cloudatlas.wmo.int/en/stratocumulus-sc.html",
            ),
            CloudInfo(
                name="Stratus",
                short="St",
                url="https://cloudatlas.wmo.int/en/stratus-st.html",
            ),
        ]

    def setup(self):
        self.ort_session = onnxruntime.InferenceSession(self.model_path)
        self.size_in = tuple(self.ort_session.get_inputs()[0].shape[2:4])
        self.name_in = self.ort_session.get_inputs()[0].name

    def infer(self, img):
        if isinstance(img, (str, io.BytesIO)):
            pil_img = Image.open(img)
        else:
            pil_img = img

        pred = self.ort_session.run(
            None, {self.name_in: self._prepea_pil_img(pil_img)}
        )
        label_idx = int(pred[0].squeeze(0).argmax())
        return label_idx

    def _prepea_pil_img(self, pil_img):
        img = pil_img.convert("RGB")
        img = img.resize(self.size_in, Image.BICUBIC)
        img_nd = np.array(img)  # h, w, ch
        img_nd = img_nd.transpose(2, 0, 1)  # ch, h, w
        img_nd = img_nd.astype("float32") / 255  # normalize
        return img_nd[None, :]
