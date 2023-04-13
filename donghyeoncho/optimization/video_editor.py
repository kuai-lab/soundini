import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils.metrics_accumulator import MetricsAccumulator
from utils.video import save_video
from optimization.raft_wrapper import RAFTWrapper
from collections import defaultdict
from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss, l1_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np
import torch.nn as nn
from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.visualization import show_tensor_image, show_editied_masked_image
import cv2
import timm
import librosa
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import os
import time
from models.facial_recognition.model_irse import Backbone
import sys
# python main2.py -p "a photo of fire, flame"  -i "samples/waves" --mask "mask_folder/waves_mask" --output_path "out_RESULT/" --mask_preservation_loss --background_preservation_loss --optical_flow_loss
# python main2.py -p "a photo of fire, flame"  -i "samples/waves" --mask "mask_folder/waves_mask" --output_path "out_RESULT/" --mask_preservation_loss --background_preservation_loss --optical_flow_loss --ddim --model_output_size 512
# python main2.py -p "beautiful water wave"  -i road_image4/ --mask water_mask_5frame --output_path "out_a_test/" --mask_preservation_loss --background_preservation_loss --optical_flow_loss --ddim --model_output_size 512
# python main2.py -p "beautiful ocean wave"  -i drought_image/ --mask drought_mask/ --output_path "out_a_test/" --mask_preservation_loss --background_preservation_loss --optical_flow_loss --ddim --model_output_size 512
# CUDA_VISIBLE_DEVICES=1 python main2.py -p "smile face"  -i ./woman  --mask ./woman_mask/ --output_path "out_a_test_woman/" --mask_preservation_loss --background_preservation_loss --optical_flow_loss --ddim --model_output_size 512

class SwinAudioEncoder(torch.nn.Module):
    def __init__(self):
        super(SwinAudioEncoder, self).__init__()
        self.feature_extractor = timm.create_model("swin_tiny_patch4_window7_224", num_classes=512, pretrained=True, in_chans=1)
        self.logit_scale_ai = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        h = self.feature_extractor(x)
        return h

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
import torchvision
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("./pretrained_models/model_ir_se50.pth"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])

            loss += 1 - diff_target
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, 
# class IDLoss2(torch.nn.Module):
#     def __init__(self):
#         super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace')
#         self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
#         self.facenet.load_state_dict(torch.load("./pretrained_models/model_ir_se50.pth"))
#         self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.cuda()
#         self.facenet.eval()

#     def extract_feats(self, x):
#         if x.shape[2] != 256:
#             x = self.pool(x)
#         # x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = x[:, :, :, :]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         return x_feats

#     def forward(self, y_hat, y):
#         n_samples = y.shape[0]
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             loss += 1 - diff_target
#             count += 1
#         return loss / count, sim_improvement / count
class VideoEditor:
    def __init__(self, args) -> None:
        self.args = args
        self.now = time.strftime('%y%m%d-%X')
        # self.args.output_path = os.path.join(self.args.output_path, self.now)#+f"_{self.args.seed}")
        if self.args.VLE_dataset:
            self.args.output_path = os.path.join(self.args.output_path)
        else:
            self.args.output_path = os.path.join(self.args.output_path, self.now+f"_{self.args.seed}")
        print("@@@ Output path: ", self.args.output_path)
        os.makedirs(self.args.output_path, exist_ok=True)
        with open(f"{self.args.output_path}/runinng_command.txt", "w") as file1:
            file1.write(' '.join(sys.argv))
        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        self.video_results_path = Path(os.path.join(self.args.output_path, "results"))
        # os.makedirs(self.ranked_results_path, exist_ok=True)
        os.makedirs(self.video_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        shape = (self.args.batch_size, 3, self.model_config["image_size"], self.model_config["image_size"])
        if self.args.frame_num == -1:
            assert len(sorted(os.listdir(self.args.init_image))) == len(sorted(os.listdir(self.args.mask)))
        if self.args.frame_num == -1:
            self.args.frame_num = len(sorted(os.listdir(self.args.init_image)))
        
        self.noise = torch.randn(*shape, device=self.device)
        self.noise = self.noise.repeat(self.args.frame_num,1,1,1)
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.clip_model = (
            clip.load("ViT-B/32", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
        # self.clip_text_model = (
        #     clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        # )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        if self.args.face:
            self.id_loss = IDLoss().to(self.device).eval()
        # self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        self.raft_wrapper = RAFTWrapper(
            model_path='thirdparty/RAFT/models/raft-things.pth', max_long_edge=780)
    def preprocess(self,img1,img2):
        # files = sorted(self.args.vid_path.glob('*.jpg'))
        #이미지 input
        # vid_name = args.vid_path.name 
        # vid_root = args.vid_path.parent
        # out_flow_dir = vid_root / f'{vid_name}_flow'
        # out_flow_dir.mkdir(exist_ok=True)
        # raft_wrapper = RAFTWrapper(
        #     model_path='thirdparty/RAFT/models/raft-things.pth', max_long_edge=780)
        # import pdb;pdb.set_trace()
        # img1 = img1.unsqueeze(0)
        flow12 = self.raft_wrapper.compute_flow(img1.unsqueeze(0), img2.unsqueeze(0))
        # flow21 = raft_wrapper.compute_flow(img2.unsqueeze(0), img1.unsqueeze(0))
        # for i, file1 in enumerate(tqdm(files,desc='computing flow')):
        #     if i < len(files) - 1:
        #         file2 = files[i + 1]
        #         fn1 = file1.name
        #         fn2 = file2.name
        #         out_flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
        #         out_flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'

        #         overwrite=False
        #         if not out_flow12_fn.exists() and not out_flow21_fn.exists() or overwrite:
        #             im1, im2 = raft_wrapper.load_images(str(file1), str(file2))
        #             flow12 = raft_wrapper.compute_flow(im1, im2)
        #             flow21 = raft_wrapper.compute_flow(im2, im1)
        #             np.save(out_flow12_fn, flow12)
        #             np.save(out_flow21_fn, flow21)
        return flow12

    def warp(self, x):
        # self.init_image_total = torch.empty(0,3,256,256).to(self.device)
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        aaaa = []
        # import pdb; pdb.set_trace()
        for i in range(B-1):
            #첫번째 이미지와 두번째 이미지 사이의 flow
            self.flow12 = self.preprocess(x[i],x[i+1])
            self.flow21 = self.preprocess(x[i+1],x[i])
            aaaa.append(self.flow12)
            aaaa.append(self.flow21)
        #optical flow 모두 모은것
        self.flow = torch.tensor(aaaa)

    
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        if x.is_cuda:
            grid = grid.cuda()
        
        # vgrid = grid[1:] + self.flow.permute(0, 3, 1, 2).cuda()
        vgrid = grid + self.flow.permute(0, 3, 1, 2).cuda()
        
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        #이미지2번부터 마지막 이미지까지 warping , img2는 img1과의 optical flow를 이용해서 img1로 warping
        # output = nn.functional.grid_sample(x[1:], vgrid)
        output = nn.functional.grid_sample(x, vgrid)
        # mask = torch.ones(x[1:].size()).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid)

        # mask[mask < 0.999] = 0
        # mask[mask > 0] = 1

        return output 


    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def clip_loss(self, x_in, text_embed, frame_num, t):
        clip_loss = torch.tensor(0)
        #마스크 부분만 뽑아서
        # import pdb; pdb.set_trace()
        if self.args.face:
            # mask_2 = self.mask_total[frame_num+1:frame_num+2]# * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(75)))
            masked_input = x_in #x_in * mask_2
        else:
            mask_2 = self.mask_total[frame_num+1:frame_num+2]
            if self.mask is not None:
                masked_input = x_in * mask_2
            else:
                masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    def edit_video_by_prompt(self):
        #text, mask, image 준비
        text_embed = self.clip_model.encode_text(
            clip.tokenize(self.args.prompt).to(self.device)
        ).float()

        #================================== 수정한 내용 ==================================#
        # audio_encoder = AudioEncoder()
        audio_encoder = SwinAudioEncoder()
        audio_encoder.load_state_dict(copyStateDict(torch.load("./pretrained_models/tpami_encoder40.pth")))
        
        audio_encoder = audio_encoder.cuda()
        audio_encoder.eval()
        audio_path = f"./audio_example/{self.args.sound_file}"
        print("audio_path:", audio_path)
        y, sr = librosa.load(audio_path, sr=44100)
        n_mels = 128
        # n_mels = 224
        time_length = 864
        audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        # audio_inputs.shape = (128, 862)
        audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
        # # audio_inputs.shape = (128, 862)
        audio_inputs = audio_inputs

        zero = np.zeros((n_mels, time_length))
        resize_resolution = 512
        # resize_resolution = 224
        h, w = audio_inputs.shape
        if w >= time_length:
            j = 0
            j = random.randint(0, w-time_length)
            audio_inputs = audio_inputs[:,j:j+time_length]
        else:
            zero[:,:w] = audio_inputs[:,:w]
            audio_inputs = zero
        # audio_inputs.shape = (128, 864)
        # audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
        audio_inputs = cv2.resize(audio_inputs, (224, 224))
        # audio_inputs.shape = (512, 128)
        audio_inputs = np.array([audio_inputs])
        # audio_inputs.shape = (1, 512, 128)
        # audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().cuda()
        audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, 224, 224))).float().cuda()
        audio_embed = audio_encoder(audio_inputs).float()
        #==============================================================================#

        # self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        # self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        # self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        # #이미지 tensor로

        # #이미지를 0~1로 바꾸고, 0~1을 -1~1로 바꾸고, 1차원 추가
        # #self.init_image = [1,3,256,256]
    
        # #frame number
        # self.frame_num = self.args.frame_num
        #폴더내 이미지 리스트
        import natsort
        if self.args.VLE_dataset:
            self.file_list = natsort.natsorted(os.listdir(self.args.init_image))
            self.file_list.insert(0, self.file_list[0])
        else:
            self.file_list = natsort.natsorted(os.listdir(self.args.init_image))
        self.path = self.args.init_image
        self.init_image_total = torch.empty(0,3,self.args.model_output_size,self.args.model_output_size).to(self.device)
        #frame size
        self.total_image_pil = []
        for k in range(self.args.frame_num):
            self.file_path = os.path.join(self.path, self.file_list[k])
            self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
            self.init_image_pil = Image.open(self.file_path).convert("RGB")
            size = self.init_image_pil.size
            # self.init_image_pil = self.init_image_pil.crop((200, 0, size[0]-200, size[1]))
            # self.init_image_pil.save("woman_face.png")
            self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.total_image_pil.append(self.init_image_pil)
            self.init_image = (
                TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
           
            self.init_image_total = torch.cat((self.init_image_total, self.init_image), 0) # [batch, 3, 256, 256]
        #####################################################################################################################################
        # self.file_list = natsort.natsorted(os.listdir("/home/lsh/blended-diffusion/test_input/kid-football_original_resized"))
        # self.path = "/home/lsh/blended-diffusion/test_input/kid-football_original_resized"
        # self.total_image_pil2 = []
        # for k in range(self.args.frame_num):
        #     self.file_path = os.path.join(self.path, self.file_list[k])
        #     self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        #     self.init_image_pil = Image.open(self.file_path).convert("RGB")
        #     size = self.init_image_pil.size
        #     # self.init_image_pil = self.init_image_pil.crop((200, 0, size[0]-200, size[1]))
        #     # self.init_image_pil.save("woman_face.png")
        #     self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        #     self.total_image_pil2.append(self.init_image_pil)
        #####################################################################################################################################
   
        
        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path)

        if self.args.VLE_dataset:
            self.mask_list = natsort.natsorted(os.listdir(self.args.mask))
            self.mask_list.insert(0, self.mask_list[0])
        else:
            self.mask_list = natsort.natsorted(os.listdir(self.args.mask))
        self.maskpath = self.args.mask
        
        self.mask_total = torch.empty(0,1,self.args.model_output_size,self.args.model_output_size).to(self.device)
        self.mask_pil = None
        self.path = self.args.mask
        self.total_mask_pil = []
        #마스크 불러오기
        if self.args.mask is not None:
            for k in range(self.args.frame_num):
                self.maskpath = os.path.join(self.path, self.mask_list[k])
                # self.maskpath = self.path + self.mask_list[k]
                self.mask_pil = Image.open(self.maskpath).convert("RGB")
                size = self.mask_pil.size
                # self.mask_pil = self.mask_pil.crop((200, 0, size[0]-200, size[1]))
                # self.mask_pil.save("woman_mask_3.png")

                # if self.mask_pil.size != self.image_size:
                #     self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
                # image_mask_pil_binarized = ((np.array(self.mask_pil) > -0.5) * 255).astype(np.uint8)
                # self.mask_pil = Image.fromarray(image_mask_pil_binarized)
                # if self.args.invert_mask:
                #     image_mask_pil_binarized = 255 - image_mask_pil_binarized
                #     self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)

                if self.mask_pil.size != self.image_size:
                    self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
                image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
                if self.args.invert_mask:
                    image_mask_pil_binarized = 255 - image_mask_pil_binarized
                    self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
                
                self.total_mask_pil.append(self.mask_pil)
                self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
                self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)
                self.mask_total = torch.cat((self.mask_total, self.mask), 0) # [batch, 1, 256, 256]

            if self.args.export_assets:
                mask_path = self.assets_path / Path(
                    self.args.output_file.replace(".png", "_mask.png")
                )
                self.mask_pil.save(mask_path)
        # 개수가 다르면
        assert self.init_image_total.shape[0] == self.mask_total.shape[0]
        #gradient func
        def cond_fn(x, t, y=None, frame_num=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_() #5,3,256,256
                t = self.unscale_timestep(t)
                # import pdb;pdb.set_trace()
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac) #5,3,256,256
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)
                
                #############################################################################################
                # Text
                if self.args.clip_guidance_lambda != 0:
                    clip_loss = self.clip_loss(x_in[1:], text_embed, frame_num, t) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                # Text + Audio
                # if self.args.clip_guidance_lambda != 0:
                #     clip_text_sound_mix_loss = self.clip_loss(x_in, text_embed*0.5 + audio_embed*0.5) * self.args.clip_guidance_lambda
                #     loss = loss + clip_text_sound_mix_loss
                #     self.metrics_accumulator.update_metric("clip_text_sound_mix_loss", clip_text_sound_mix_loss.item())
                # Audio
                if self.args.sound_guidance_lambda != 0:
                    # print("곱해지는 값", self.args.sound_guidance_lambda * ((((frame_num) / self.args.frame_num)*0.3)+0.6))
                    sound_clip_loss = self.clip_loss(x_in[1:], audio_embed, frame_num, t) * self.args.sound_guidance_lambda# * ((((frame_num) / self.args.frame_num)*0.3)+0.6)
                    loss = loss + sound_clip_loss
                    self.metrics_accumulator.update_metric("sound_clip_loss", sound_clip_loss.item())
                #############################################################################################
                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"][1:]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                #================================== 수정한 내용 ==================================#
                if self.args.face:
                    if False: #self.mask is not None:
                        masked_face = x_in * self.mask_total[frame_num:frame_num+1]
                    else:
                        masked_face = x_in[1:]
                    # init_mask_image = self.init_image * (self.mask)
                    # import pdb; pdb.set_trace()
                    # original_image_masked = self.init_image_total[frame_num:frame_num+1] * self.mask_total[frame_num:frame_num+1]
                    id_loss_val = self.id_loss(masked_face, self.init_image_total[frame_num+1:frame_num+2])[0]
                    loss = loss + id_loss_val
                    self.metrics_accumulator.update_metric("id_loss", id_loss_val.item())
                # ================================================================================#
                # if True:
                    # if self.mask is not None:
                    #     masked_face = x_in * self.mask_total[frame_num:frame_num+1]
                    # else:
                    #     masked_face = x_in
                    # # init_mask_image = self.init_image * (self.mask)
                    # # import pdb; pdb.set_trace()
                    # id_loss_prev_val = self.id_loss(masked_face[:1], masked_face[1:2])[0] * 1000
                    # loss = loss + id_loss_prev_val
                    # self.metrics_accumulator.update_metric("id_loss_prev", id_loss_prev_val.item())  
                #================================================================================#
                if self.args.background_preservation_loss:
                    if False:#self.mask is not None:
                        masked_background = x_in[1:] * (1 - self.mask_total[frame_num+1:frame_num+2])
                    else:
                        masked_background = x_in[1:]
                    if self.args.lpips_sim_lambda:
                        lpips_loss = self.lpips_model(masked_background, self.init_image_total[frame_num+1:frame_num+2]).sum() * self.args.lpips_sim_lambda * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(50)))
                        loss = loss + lpips_loss
                    if self.args.l2_sim_lambda:
                        l1_sim_lambda_loss = l1_loss(masked_background, self.init_image_total[frame_num+1:frame_num+2]) * self.args.l2_sim_lambda * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(50)))
                        loss = loss + l1_sim_lambda_loss
                    self.metrics_accumulator.update_metric("background_preservation_loss", lpips_loss.item()+l1_sim_lambda_loss.item())

                # if self.args.mask_preservation_loss:
                #     if self.mask is not None:
                #         masked_sky = x_in * (self.mask_total[frame_num:frame_num+2])
                #     else:
                #         masked_sky = x_in
                #     mask_preservation_loss = mse_loss(masked_sky[0], masked_sky[1]) * 1000.0
                #     loss = loss + mask_preservation_loss
                #     self.metrics_accumulator.update_metric("mask_preservation_loss", mask_preservation_loss.item())
                
                # optical flow = f12를 이용해서 im2를 im1에 warp = im2를 im1 시점으로 나타내기 = 이론적으로 im2 와 im1이 같다.
                if self.args.optical_flow_loss != 0:
                    # mask_2 = self.mask_total[frame_num:frame_num+2] * ((75-t[0])/75)
                    if self.mask is not None:
                        masked_sky = x_in * self.mask_total[frame_num:frame_num+2]
                    else:
                        masked_sky = x_in
                    
                    #warping
                    warp_output = self.warp(masked_sky)
                    # 와핑한 img_hat1, img_hat2, img_hat3와 img2, img3, img4의 mse loss를 구함
                    # print("곱해지는 값", min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(self.diffusion.num_timesteps/2))))
                    optical_flow_loss1 = mse_loss(warp_output[0],masked_sky[1]) * self.args.optical_flow_loss * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(50)))
                    optical_flow_loss2 = mse_loss(warp_output[1],masked_sky[0]) * self.args.optical_flow_loss * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(50)))
                    optical_flow_loss = optical_flow_loss1 + optical_flow_loss2
                    loss = loss + optical_flow_loss
                    self.metrics_accumulator.update_metric("optical_flow_loss", optical_flow_loss.item())
                return -torch.autograd.grad(loss, x)[0]

        # local clip-guided diffusion에서는 사용하지 않고 text-driven blended diffusion에서 사용
        @torch.no_grad()
        def postprocess_fn(out, t, frame_num):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image_total[frame_num:frame_num+2], t)
                #마스크 있는 부분과 없는 부분 
                # mask_2 = self.mask_total[frame_num:frame_num+2] * min(1.0, ((75-t[0].item())/40))
                # print("곱하는 값:", min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(50))))
                mask_2 = self.mask_total[frame_num:frame_num+2]# * min(1.0, (((self.diffusion.num_timesteps - self.args.skip_timesteps)-t[0].item())/(self.args.original_guidance)))
                out["sample"] = out["sample"] * mask_2 + background_stage_t * (1 - mask_2)
            
                
            return out

        save_image_interval = 10#self.diffusion.num_timesteps // 10
        for iteration_number in range(self.args.iterations_num):#8
            print(f"Start iterations {iteration_number}")
            #diffusion model
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            #samples :
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([2], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image_total, 
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True,
                noise=self.noise,
                frame_num=self.args.frame_num,
                random_seed=self.args.seed,
                seed_fix_index=self.args.seed_fix_index,
            )

            intermediate_samples = [[] for i in range(self.args.frame_num)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    # os.makedirs(os.path.join(self.args.output_path, 'results', f"{j}"), exist_ok=True)
                    self.metrics_accumulator.print_average_metric()
                    for sam in range(len(sample)):
                        # for b in range(2):
                        # 마지막 frame 이면
                        # if sam == (len(sample))-1:
                        #     pred_image = sample[sam]["pred_xstart"][1]
                        # else:
                        pred_image = sample[sam]["pred_xstart"][0]
                        if sam == len(sample)-1:
                            pred_image = sample[sam]["pred_xstart"][1]
                        # pred_image = sample[sam]["sample"][0]
                        # if sam == len(sample)-1:
                        #     pred_image = sample[sam]["sample"][1]
                        
                        # import pdb;pdb.set_trace()
                    
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        frame_num_save = sam
                        visualization_path = visualization_path.with_stem(
                            f"frame_{frame_num_save:02d}"
                        )
                    
                        #not local clip-guided diffusion
                        if (
                            self.mask is not None
                            and self.args.enforce_background
                            and j == total_steps
                            and not self.args.local_clip_guided_diffusion
                        ):
                            pred_image = (
                                self.init_image_total[frame_num_save] * (1 - self.mask_total[frame_num_save]) + pred_image * self.mask_total[frame_num_save]
                            )
                        
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)#0~1사이로
                        #tensor -> pil 이미지 타입 변경
                        pred_image_pil = TF.to_pil_image(pred_image)
                        #마스크 부분 pred image 잘라서 텍스트와 차이 구하기, 비교
                        masked_pred_image = self.mask_total[frame_num_save] * pred_image.unsqueeze(0)
                        final_distance = self.unaugmented_clip_distance(
                            masked_pred_image, text_embed
                        )
                        formatted_distance = f"{final_distance:.4f}"

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        # if j == total_steps:
                        #     path_friendly_distance = formatted_distance.replace(".", "")
                        #     ranked_pred_path = self.ranked_results_path / (
                        #         path_friendly_distance + "_" + visualization_path.name
                        #     )
                        #     pred_image_pil.save(ranked_pred_path)
                        if j == total_steps:
                            video_pred_path = self.video_results_path / (
                                visualization_path.name
                            )
                            pred_image_pil = pred_image_pil#.resize((512,512))
                            pred_image_pil.save(video_pred_path)
                        # if True:
                        #     video_pred_path = self.video_results_path / (
                        #         f"{j}"
                        #     )
                        #     video_pred_path = video_pred_path / (
                        #         visualization_path.name
                        #     )
                        #     pred_image_pil = pred_image_pil#.resize((512,512))
                        #     pred_image_pil.save(video_pred_path)

                        # intermediate_samples[sam].append(pred_image_pil)
                        if True:
                            show_editied_masked_image(
                                title=self.args.prompt,
                                source_image=self.total_image_pil[frame_num_save],
                                edited_image=pred_image_pil,
                                mask=self.total_mask_pil[frame_num_save],
                                path=visualization_path,
                                distance=formatted_distance,
                            )
            print('end')
            # if self.args.save_video:
            # if True:
            #     for b in range(self.args.frame_num):
            #         video_name = self.args.output_file.replace(
            #             ".png", f"_i_{iteration_number}_b_{b}.avi"
            #         )
            #         video_path = os.path.join(self.args.output_path, video_name)
            #         save_video(intermediate_samples[b], video_path)

    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256
            else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)
