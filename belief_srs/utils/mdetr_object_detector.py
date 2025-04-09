import logging
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours

torch.set_grad_enabled(False);

logger = logging.getLogger(__name__)

class MDETRObjectDetector():
    """
    Detects handles and returns a bounding box around the most likely location.
    """

    def __init__(self, debug=False):
        self.debug = debug
        model, _ = torch.hub.load(
            'ashkamath/mdetr:main', 'mdetr_efficientnetB5',
            pretrained=True, return_postprocessor=True)
        model = model.cuda()
        model.eval();
        self.model = model

        # self.caption = "wood door with metal handle"
        # thumb-turn
        # self.caption = "wood door with thumbturn"
        # self.caption = "wood door with deadbolt knob"
        self.caption = "wood door with small metal tab above handle"
        # self.caption = "wood door with deadbolt"
        # self.caption = "wood door with metal handle and thumbturn"
        # self.caption = "wood door with metal handle and thumb turn"
        # self.caption = "wood door with metal handle and deadbolt"
        # self.caption = "wood door with lever handle"
        # self.caption = "door. metal handle"
        # self.caption = "metal handle"
        # self.caption = "door. metal handle"
        # self.caption = "door with handle"

    def detect(self, image, object_str="handle", caption=None, **kwargs):
        if type(image) is not Image.Image:
            image = Image.fromarray(image)

        if caption is None:
            caption = self.caption
        labels, bboxes, probas = self._detect(image, caption, **kwargs)
        logger.debug(f"  Labels: {labels}")

        obj_labels, obj_bboxes, obj_probas = [], [], []
        for i, label in enumerate(labels):
            if object_str in label:
                obj_labels.append(label.strip())
                obj_bboxes.append(bboxes[i].numpy())
                obj_probas.append(probas[i].numpy())
        obj_probas, obj_bboxes = np.array(obj_probas), np.array(obj_bboxes)

        if len(obj_probas):
            best_detection = np.argmax(obj_probas)
        else:
            return None, None, None

        return obj_bboxes[best_detection], obj_labels[best_detection], obj_probas[best_detection]

    def _detect(self, image, caption, **kwargs):
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(image).unsqueeze(0).cuda()

        logger.debug(f"  Detecting {caption}")
        # propagate through the model
        memory_cache = self.model(img, [caption], encode_and_save=True)
        outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        # keep only predictions with 0.7+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.7).cpu()
        logger.debug(f"  Keep: {keep}")

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(
            outputs['pred_boxes'].cpu()[0, keep], image.size)

        # Extract the text spans predicted by each box
        positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero(as_tuple=False).tolist()
        predicted_spans = defaultdict(str)
        for tok in positive_tokens:
            item, pos = tok
            if pos < 255:
                span = memory_cache["tokenized"].token_to_chars(0, pos)
                predicted_spans [item] += " " + caption[span.start:span.end]

        labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
        if self.debug:
            self.plot_results(image, probas[keep], bboxes_scaled, labels, **kwargs)

        return labels, bboxes_scaled, probas[keep]

    def plot_results(self, img, scores, boxes, labels, masks=None, ax=None, animated=False):
        if type(img) is Image.Image:
            np_image = np.array(img)
        else:
            np_image = np.array(img)

        if ax is None:
            fig = plt.figure(figsize=(16,10))
            ax = fig.gca()
        ax.clear()
        colors = self.COLORS * 100
        if masks is None:
            masks = [None for _ in range(len(scores))]
        assert len(scores) == len(boxes) == len(labels) == len(masks)
        for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            text = f'{l}: {s:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

            if mask is None:
                continue
            np_image = self.apply_mask(np_image, mask, c)

            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=c)
                ax.add_patch(p)

        plt.axis('off')
        ax_img = plt.imshow(np_image, animated=animated)
        return ax_img

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

        # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    @staticmethod
    def apply_mask(image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    # for output bounding box post-processing
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
