from typing import Any, Dict, List, Optional
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import v2
import numpy as np
from torchvision.utils import draw_keypoints
import IB_DL.utils as utils
from IB_DL.utils import cprint
from IB_DL.datasets import Loop


class TensorTransforms(v2.Transform):
    def __init__(self):
        super().__init__()

    def _check_input(self, inpt: torch.FloatTensor):
        if not isinstance(inpt, torch.FloatTensor):
            raise TypeError(f"Expected input to be a tensor, got {type(inpt)}")


class ImageTransforms(v2.Transform):
    def __init__(self):
        super().__init__()

    def _check_input(self, inpt:  torch.FloatTensor):
        if not isinstance(inpt, torch.FloatTensor):
            raise TypeError(
                f"Expected input to be an floattensor, got {type(inpt)}")


class MedianLoops(v2.Transform):
    """
    Computes the median of a list of Loop objects.
    """

    def __call__(self, inpt: List[Any], *args: Any, **kwargs: Any) -> Any:
        """
        Computes the median of a list of Loop objects.
        Args:
            inpt (List[Loop]): List of Loop objects.
            params (Dict[str, Any]): Additional parameters (not used).
        Returns:
            Loop: A new Loop object with the median values of volume, flow, resistance, and reactance.
        """
        V_median = np.median(np.array([loop.volume for loop in inpt]),
                             axis=0).tolist()
        F_median = np.median(np.array([loop.flow for loop in inpt]),
                             axis=0).tolist()
        R_median = np.median(np.array([loop.resistance for loop in inpt]),
                             axis=0).tolist()
        X_median = np.median(np.array([loop.reactance for loop in inpt]),
                             axis=0).tolist()
        median_loop = Loop(inpt[0].patient_id, "-1",
                           V_median, F_median, R_median, X_median)

        return median_loop, *args

    def _check_input(self, inpt: List[Loop]):
        if not isinstance(
                inpt, list) or not all(
                isinstance(item, Loop) for item in inpt):
            raise TypeError(f"Expected input to be a tensor, got {type(inpt)}")


class LoopToTensor(v2.Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, inpt: Loop, *args: Any, **kwargs: Any) -> Any:
        """
        Converts a Loop object to a torch FloatTensor.
        This tensor contains the flow, resistance, and reactance data of the loop.

        Args:
            loop (Loop): Instance of the Loop class containing flow, resistance, and reactance data.

        Returns:
            torch.FloatTensor: size (4, 50) tensor containing flow, resistance, and reactance data respectively.
        """
        loop_data = np.stack(
            (inpt.flow, inpt.volume, inpt.resistance, inpt.reactance), axis=0)
        return torch.FloatTensor(loop_data), *args

    def _check_input(self, inpt: Loop):
        if not torch.is_tensor(inpt):
            raise TypeError(f"Expected input to be a tensor, got {type(inpt)}")


class ToImage(TensorTransforms):
    def __init__(self,
                 img_size: int = 128,
                 linewidth: int = 2,
                 img_channels: list[str] = ["f;abs", "f;abs", "f;abs"],
                 num_skips: int = 0,
                 max_len_skips: int = 0,
                 change_lwidth: bool = False):
        """A transformation that converts the input loop tensor into an image tensor by plotting flow/volume against r/x with varying options for scaling and visualization. Expects input shape (C=4, L) and outputs a tensor of shape (C_img, img_size, img_size) where C_img is determined by img_channels.

        Args:
            img_size (int, optional): The height and width (in pixels) of the output square image tensor. Defaults to 128.
            linewidth (int, optional): The base thickness of the lines drawn to connect the data points on the canvas. Defaults to 2.
            img_channels (list[str], optional): Configuration for each image channel formatted as "x_axis;scale_mode" (e.g., 'f;abs' for Flow with absolute scaling, or 'v;rel' for Volume with relative scaling). Supported scale modes are 'abs', 'rel', and 'zc'. Defaults to ["f;abs", "f;abs", "f;abs"].
            num_skips (int, optional): Data augmentation parameter defining the number of random segments to erase (skip) from the drawn loops. Defaults to 0.
            max_len_skips (int, optional): Data augmentation parameter defining the maximum length (in number of points) of each erased segment. Defaults to 0.
            change_lwidth (bool, optional): Data augmentation parameter. If True, applies a periodic variation to the line thickness (1x -> 2x -> 1x -> 0.5x -> 1x) to simulate visual distortion. Defaults to False.
        """
        super().__init__()
        self.img_size = img_size
        self.linewidth = linewidth
        self.img_channels = img_channels
        self.num_skips = num_skips
        self.max_len_skips = max_len_skips
        self.change_lwidth = change_lwidth

    def __call__(self, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        if not args[0]["augment"]:
            num_skips = 0
            max_len_skips = 0
        else:
            num_skips = self.num_skips
            max_len_skips = self.max_len_skips

        args[0]["fsclae"] = (torch.max(inpt[[0], :])-torch.min(inpt[[0], :])
                             )/(utils.F_MAX-utils.F_MIN)
        args[0]["vsclae"] = (torch.max(inpt[[1], :])-torch.min(inpt[[1], :])
                             )/(utils.V_MAX-utils.V_MIN)
        args[0]["rsclae"] = (torch.max(inpt[[2], :])-torch.min(inpt[[2], :])
                             )/(utils.R_MAX-utils.R_MIN)
        args[0]["xsclae"] = (torch.max(inpt[[3], :])-torch.min(inpt[[3], :])
                             )/(utils.X_MAX-utils.X_MIN)

        pos_mask = inpt[0, :] > 0
        neg_mask = inpt[0, :] < 0
        pos_mean = inpt[2, pos_mask].mean() if pos_mask.any() else 0
        neg_mean = inpt[2, neg_mask].mean() if neg_mask.any() else 0
        args[0]["rrotation"] = (pos_mean - neg_mean)
        pos_mean = inpt[3, pos_mask].mean() if pos_mask.any() else 0
        neg_mean = inpt[3, neg_mask].mean() if neg_mask.any() else 0
        args[0]["xrotation"] = (pos_mean - neg_mean)

        # empty black canvas, RGB for torchvision
        my_canvas = torch.zeros(
            (len(self.img_channels), self.img_size, self.img_size), dtype=torch.float32)

        # zero flow and zero Zrs is in the middle of the canvas
        for idx, channel in enumerate(self.img_channels):
            tmp_canvas = torch.zeros(
                (3, self.img_size, self.img_size), dtype=torch.float32)

            if channel.split(';')[0] == 'f':
                x_min = utils.F_MIN
                x_max = utils.F_MAX
                x_axes_data = inpt[0, :]
            else:
                x_min = utils.V_MIN
                x_max = utils.V_MAX
                x_axes_data = inpt[1, :]

            if channel.split(';')[1] == 'abs':
                y_min = utils.X_MIN
                y_max = utils.R_MAX

                f_scaled = self.img_size * \
                    (x_axes_data - x_min) / (x_max-x_min)
                r_scaled = self.img_size * (inpt[2, :] - y_min) / (y_max-y_min)
                x_scaled = self.img_size * (inpt[3, :] - y_min) / (y_max-y_min)
            elif channel.split(';')[1] == 'rel':
                y_min = torch.min(inpt[[2, 3], :])
                y_max = torch.max(inpt[[2, 3], :])
                x_max = torch.max(x_axes_data)
                x_min = torch.min(x_axes_data)

                f_scaled = (x_axes_data - x_min) / \
                    (x_max - x_min) * self.img_size
                r_scaled = (inpt[2, :] - y_min) / \
                    (y_max - y_min) * self.img_size
                x_scaled = (inpt[3, :] - y_min) / \
                    (y_max - y_min) * self.img_size
            elif channel.split(';')[1] == 'zc':
                x_diff = torch.max(torch.abs(x_axes_data))
                y_diff = torch.max(torch.abs(inpt[[2, 3], :]))

                f_scaled = x_axes_data * \
                    (self.img_size / 2 / x_diff) + self.img_size / 2
                r_scaled = inpt[2, :] * \
                    (self.img_size / 2 / y_diff) + self.img_size / 2
                x_scaled = inpt[3, :] * \
                    (self.img_size / 2 / y_diff) + self.img_size / 2
            else:
                raise ValueError(f"Unknown layer type: {channel}")

            r_points = torch.stack((f_scaled, r_scaled), dim=1)
            x_points = torch.stack((f_scaled, x_scaled), dim=1)
            keypoints = torch.stack(
                (r_points, x_points)).round().to(torch.int32)

            r_connections = [(i, i+1) for i in range(len(r_points) - 1)]
            r_connections.append((len(r_points) - 1, 0))
            x_connections = [(i, i+1) for i in range(len(x_points) - 1)]
            x_connections.append((len(x_points) - 1, 0))
            my_connections = r_connections + x_connections

            visibility = torch.ones(keypoints.shape[:2], dtype=torch.bool)
            for i in range(keypoints.shape[0]):
                for j in range(keypoints.shape[1]):
                    x, y = keypoints[i, j]
                    x = x.int()
                    y = y.int()
                    if x < 0 or x >= self.img_size or y < 0 or y >= self.img_size:
                        visibility[i, j] = False

            skip_starts = np.random.randint(
                0, keypoints.shape[0]+keypoints.shape[1], num_skips)
            for skip_start in skip_starts:
                skip_end = skip_start + \
                    np.random.randint(1, max_len_skips + 1)
                channel_len = keypoints.shape[1]
                if skip_start < channel_len:
                    # First channel
                    indices = np.arange(skip_start, skip_end) % channel_len
                    visibility[0, indices] = False
                else:
                    # Second channel
                    indices = np.arange(
                        skip_start - channel_len, skip_end - channel_len) % channel_len
                    visibility[1, indices] = False

            # Draw connections with varying linewidths: 1*lw -> 2*lw -> 1*lw -> 0.5*lw -> 1*lw
            if self.change_lwidth:
                # a periodic pattern of linewidths, with a random starting offset
                linewidths = [self.linewidth, 2*self.linewidth,
                              self.linewidth, 0.5*self.linewidth, self.linewidth]
                num_connections = len(my_connections)
                offset = np.random.randint(0, len(linewidths))
                for i, (start_idx, end_idx) in enumerate(my_connections):
                    lw = linewidths[(i + offset) % len(linewidths)]
                    # Determine which channel (0 or 1) this connection belongs to
                    channel_idx = 0 if i < len(r_connections) else 1
                    # Only draw if both points are visible
                    if visibility[channel_idx, start_idx] and visibility[channel_idx, end_idx]:
                        pt1 = keypoints[channel_idx, start_idx]
                        pt2 = keypoints[channel_idx, end_idx]
                        tmp_canvas = draw_keypoints(
                            image=tmp_canvas,
                            keypoints=torch.stack([pt1, pt2]).unsqueeze(0),
                            connectivity=[(0, 1)],
                            colors="white",
                            radius=0,
                            width=int(lw),
                            visibility=torch.ones((1, 2), dtype=torch.bool)
                        )
            else:
                tmp_canvas = draw_keypoints(
                    image=tmp_canvas,
                    keypoints=keypoints,
                    connectivity=my_connections,
                    colors="white",
                    radius=0,
                    width=self.linewidth,
                    visibility=visibility
                )
            my_canvas[idx, :, :] = tmp_canvas[0, :, :]

        # Flip vertically (height axis)
        my_canvas = torch.flip(my_canvas, dims=[1])

        return my_canvas, *args

    def __repr__(self):
        return f"{self.__class__.__name__}(absolute scale={self.abs_scale})"


class FlattenTensor(TensorTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        return torch.flatten(inpt), *args


class AWG(TensorTransforms):
    def __init__(self, max_mu: float = 0.0, sigma: float = 1.0, p: float = 0.5):
        """
        Transform that adds Additive White Gaussian Noise (AWGN) to the input tensor.
        Args:
            max_mu (float): Maximum mean value for the Gaussian noise.
            sigma (float): Standard deviation for the Gaussian noise.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        self.max_mu = max_mu
        self.sigma = sigma
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        if not args[0]["augment"] or np.random.rand() > self.p:
            return inpt, *args
        mu = np.random.uniform(0, self.max_mu)
        noise = torch.normal(mean=mu, std=self.sigma, size=inpt.shape)
        # do not add noise to the first two channel (flow, volume)
        noise[0:2, :] = torch.zeros_like(noise[0:1, :])
        return inpt + noise, *args


class SaltAndPepperNoise(TensorTransforms):
    def __init__(self, salt_prob: float = 0.01, pepper_prob: float = 0.01,
                 max_amplitude: float = 1.0, p: float = 0.5):
        """A transformation that adds salt and pepper noise to the input tensor by randomly adding or subtracting a specified amplitude to random elements in the tensor, simulating the effect of salt and pepper noise. The noise is only applied to the last two channels (r and x) to preserve the integrity of flow and volume data.

        Args:
            salt_prob (float, optional): Probability of adding salt noise. Defaults to 0.01.
            pepper_prob (float, optional): Probability of adding pepper noise. Defaults to 0.01.
            max_amplitude (float, optional): Maximum amplitude of the noise. Defaults to 1.0.
            p (float, optional): Probability of applying the transformation. Defaults to 0.5.
        """
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.max_amplitude = max_amplitude
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        if not args[0]["augment"]:
            return inpt, *args

        amplitude = np.random.uniform(0, self.max_amplitude)
        noise = torch.rand(inpt.shape)
        noise[0:2, :] = torch.zeros_like(noise[0:1, :]) + 0.5
        if np.random.rand() < self.p:
            inpt[noise < self.salt_prob] += amplitude  # Salt
            inpt[noise > (1 - self.pepper_prob)] -= amplitude  # Pepper
        return inpt, *args


class StretchLoops(TensorTransforms):
    def __init__(self, max_delta_r: float = 0.1, max_num_of_axes: int = 1, p: float = 0.5):
        """
        A transformation that randomly stretches or compresses the loop along one or more axes (flow, volume, r, x) by a random factor within a specified range, simulating variations in the loop shapes. The transformation selects a random subset of axes to modify and applies a scaling factor that can either increase or decrease the values along those axes.

        Args:
            max_delta_r (float): Maximum relative change in the selected axis.
            max_num_of_axes (int): Maximum number of axes to be modified.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        assert max_num_of_axes > 0, "max_num_of_axes must be greater than 0"
        assert max_num_of_axes <= 4, "max_num_of_axes must be less than or equal to 4"
        self.max_delta_r = max_delta_r
        self.max_num_of_axes = max_num_of_axes
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        if not args[0]["augment"] or np.random.rand() > self.p:
            return inpt, *args

        num_of_modified_axes = np.random.randint(1, self.max_num_of_axes + 1)
        modified_axes = np.random.choice(
            [0, 1, 2, 3], size=num_of_modified_axes, replace=False)
        for axis in modified_axes:
            median_value = torch.median(inpt[axis, :])
            delta_r = np.random.uniform(0.1*self.max_delta_r, self.max_delta_r)
            inpt[axis, :] = torch.where(
                inpt[axis, :] > median_value,
                inpt[axis, :] * (1 + delta_r),
                inpt[axis, :] * (1 - delta_r)
            )

        return inpt, *args


class ShiftLoops(TensorTransforms):
    def __init__(self, max_shift: float = 0.1, max_num_of_axes: int = 1, p: float = 0.5):
        """
        A transformation that randomly shifts the loop along one or more axes (flow, volume, r, x) by a random factor within a specified range, simulating variations in the loop positions. The transformation selects a random subset of axes to modify and applies a shift factor that can either increase or decrease the values along those axes.

        Args:
            max_shift (float): Maximum relative shift value.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        assert max_num_of_axes > 0, "max_num_of_axes must be greater than 0"
        assert max_num_of_axes <= 4, "max_num_of_axes must be less than or equal to 4"
        self.max_shift = max_shift
        self.max_num_of_axes = max_num_of_axes
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        if not args[0]["augment"] or np.random.rand() > self.p:
            return inpt, *args

        num_of_modified_axes = np.random.randint(1, self.max_num_of_axes + 1)
        modified_axes = np.random.choice(
            [0, 1, 2, 3], size=num_of_modified_axes, replace=False)
        for axis in modified_axes:
            max_value = torch.max(torch.abs(inpt[axis, :]))
            shift_value = float(np.random.uniform(-self.max_shift * max_value,
                                                  self.max_shift * max_value))
            inpt[axis, :] += shift_value

        return inpt, *args


class ElasticImageTransform(ImageTransforms):
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.5):
        """
        A transformation that applies elastic deformation to the input image tensor, simulating realistic distortions in the loop shapes. The transformation uses a random displacement field generated by applying Gaussian smoothing to random noise, which is then applied to the pixel coordinates of the image to create a warped version of the original image.

        Args:
            alpha (float): Scaling factor for the deformation.
            sigma (float): Smoothing factor for the deformation.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        self.elastic = v2.ElasticTransform(alpha=alpha, sigma=sigma)
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        self._check_input(inpt)
        if not args or not args[0].get("augment", False) or np.random.rand() > self.p:
            return inpt, *args
        return self.elastic(inpt), *args


class RandomRotatiion(ImageTransforms):
    def __init__(self, degree: float = 10, p: float = 0.5):
        """
        A transformation that applies a random rotation to the input image tensor by a random angle within a specified range, simulating variations in the loop orientations. The transformation rotates the image around its center by an angle randomly selected from a uniform distribution between -degree and +degree.

        Args:
            degree (float): Maximum rotation angle in degrees.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        self.rotation = v2.RandomRotation(degrees=degree)  # type: ignore
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        self._check_input(inpt)
        if not args or not args[0].get("augment", False) or np.random.rand() > self.p:
            return inpt, *args
        return self.rotation(inpt), *args


class GaussianBlur(ImageTransforms):
    def __init__(
            self, kernel_size: tuple = (5, 9),
            sigma: tuple = (0.1, 5),
            p: float = 0.5):
        """
        A transformation that applies Gaussian blur to the input image tensor, simulating variations in the loop sharpness and introducing a smoothing effect. The transformation uses a Gaussian kernel with a specified size and standard deviation to blur the image, which can help improve model robustness by making it less sensitive to small details in the loop shapes.

        Args:
            kernel_size (tuple): Size of the Gaussian kernel.
            sigma (tuple): Standard deviation for Gaussian kernel.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        self.gaussianblur = v2.GaussianBlur(
            kernel_size=kernel_size, sigma=sigma)  # type: ignore
        self.p = p

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        self._check_input(inpt)
        if not args or not args[0].get("augment", False) or np.random.rand() > self.p:
            return inpt, *args
        return self.gaussianblur(inpt), *args


class AddScale(ImageTransforms):
    def __init__(self, num_extra_features: int = 6):
        """
        A transformation that appends extra scalar parameters (yscale, xscale, rscale, fscale, and optionally rotations) as constant feature maps to the input image tensor, allowing the model to accept a single 4D tensor (B, C_total, H, W) that includes both the original image channels and the additional scalar information. The transformation expects the input tensor to have a shape of (C_img, H, W) and relies on the presence of specific keys in the args dictionary (e.g., args[0]['ysclae'], args[0]['xsclae'], etc.) that are set by the ToImage transformation.

        Args:
            kernel_size (tuple): Size of the Gaussian kernel.
            sigma (tuple): Standard deviation for Gaussian kernel.
            p (float): Probability of applying the transformation.
        """
        super().__init__()
        self.num_extra_features = num_extra_features
        assert num_extra_features in [
            4, 6], "num_extra_features must be either 4 or 6 (for including rotations)"

    def __call__(self, inpt: torch.FloatTensor, *args: Any, **kwargs: Any):
        """
        Append extra scalar params (ysclae, xsclae) as constant feature maps so the model
        can accept a single 4D tensor (B, C_total, H, W). Expects inpt shape (C_img, H, W)
        and args to contain dicts where args[1]['ysclae'] and args[2]['xsclae'] were set by ToImage.
        Returns a tensor (C_img + 2, H, W), *args.
        """
        self._check_input(inpt)

        try:
            f_scale = args[0]["fsclae"]
            v_scale = args[0]["vsclae"]
            r_scale = args[0]["rsclae"]
            x_scale = args[0]["xsclae"]
            r_rotation = args[0]["rrotation"]
            x_rotation = args[0]["xrotation"]
        except Exception as e:
            raise ValueError(
                "Expected args[1]['ysclae'] and args[2]['xsclae'] to be present") from e

        try:
            f_val = float(f_scale)
            v_val = float(v_scale)
            r_val = float(r_scale)
            x_val = float(x_scale)
            rrot_val = float(r_rotation)
            xrot_val = float(x_rotation)
        except Exception as e:
            raise TypeError(
                "ysclae/xsclae must be convertible to float") from e

        # inpt expected shape: (C_img, H, W)
        if inpt.dim() != 3:
            raise ValueError(
                f"Expected image tensor with 3 dims (C,H,W), got {inpt.shape}")

        _, H, W = inpt.shape
        device = inpt.device
        dtype = inpt.dtype

        extra_maps = torch.empty(
            (self.num_extra_features, H, W),
            device=device, dtype=dtype)
        extra_maps[0].fill_(f_val)
        extra_maps[1].fill_(v_val)
        extra_maps[2].fill_(r_val)
        extra_maps[3].fill_(x_val)
        if self.num_extra_features == 6:
            extra_maps[4].fill_(rrot_val)
            extra_maps[5].fill_(xrot_val)

        combined = torch.cat((inpt, extra_maps), dim=0)
        return combined, *args


def build_transformations(params: Dict) -> v2.Compose:
    """
    Builds a sequence of transformations based on the provided parameters.
    Options include MedianLoops, LoopToTensor, ToImage, and Grayscale.

    Args:
        params (Dict): Dictionary containing transformation parameters.
    Returns:   A sequence of transformations wrapped in v2.Compose.
    """

    transform_objs = []
    for t in params["transformations"]:
        if t["name"] == "MedianLoops":
            transform_objs.append(MedianLoops())
        elif t["name"] == "LoopToTensor":
            transform_objs.append(LoopToTensor())
        elif t["name"] == "ToImage":
            transform_objs.append(ToImage(
                img_size=t["img_size"],
                linewidth=t["linewidth"],
                img_channels=t["img_channels"],
                num_skips=t.get("num_skips", 0),
                max_len_skips=t.get("max_len_skips", 0),
                change_lwidth=t.get("change_lwidth", False)
            ))
        elif t["name"] == "AWG":
            transform_objs.append(AWG(
                max_mu=t["max_mu"],
                sigma=t["sigma"],
                p=t["p"]
            ))
        elif t["name"] == "SaltAndPepperNoise":
            transform_objs.append(SaltAndPepperNoise(
                salt_prob=t["salt_prob"],
                pepper_prob=t["pepper_prob"],
                max_amplitude=t["max_amplitude"],
                p=t["p"]))
        elif t["name"] == "StretchLoops":
            transform_objs.append(StretchLoops(
                max_delta_r=t["max_delta_r"],
                max_num_of_axes=t["max_num_of_axes"],
                p=t["p"]))
        elif t["name"] == "ShiftLoops":
            transform_objs.append(ShiftLoops(
                max_shift=t["max_shift"],
                max_num_of_axes=t["max_num_of_axes"],
                p=t["p"]))
        elif t["name"] == "Grayscale":
            transform_objs.append(v2.Grayscale())
        elif t["name"] == "Flatten":
            transform_objs.append(FlattenTensor())
        elif t["name"] == "ElasticTransform":
            transform_objs.append(ElasticImageTransform(
                alpha=t["alpha"],
                sigma=t["sigma"],
                p=t["p"]
            ))
        elif t["name"] == "RandomRotation":
            transform_objs.append(RandomRotatiion(
                degree=t["degree"],
                p=t["p"]
            ))
        elif t["name"] == "GaussianBlur":
            transform_objs.append(GaussianBlur(
                kernel_size=t["kernel_size"],
                sigma=t["sigma"],
                p=t["p"]
            ))
        elif t["name"] == "AddScale":
            transform_objs.append(AddScale())
        else:
            raise ValueError(f"Unknown transformation: {t['name']}")  # type: ignore

    return v2.Compose(transform_objs)
