import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import open3d as o3d

from src.utils.fmm.depth_utils import (
    get_point_cloud_from_z_t,
    transform_camera_view_t,
    transform_pose_t,
    splat_feat_nd
)
from src.utils.model import get_grid, ChannelPool
from src.utils.camera import get_camera_matrix


class Mapping(nn.Module):
    def __init__(self, args):
        super(Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height # 160
        self.screen_w = args.frame_width # 120
        self.resolution = args.map_resolution # 5 | size of the voxel 5 centimeter
        self.z_resolution = args.map_resolution # 5
        self.map_size_cm = args.map_size_cm // args.global_downscaling # 3600 // 3
        self.n_channels = 3
        self.vision_range = args.vision_range # default 100 | 100 voxels
        self.dropout = 0.5
        self.fov = args.hfov # default 79
        self.du_scale = args.du_scale # downscale default 1
        self.cat_pred_threshold = args.cat_pred_threshold # 5.0
        self.exp_pred_threshold = args.exp_pred_threshold # 1.0
        self.map_pred_threshold = args.map_pred_threshold # 1.0
        self.num_sem_categories = args.num_sem_categories # default 16

        self.max_height = int(360 / self.z_resolution) # 72
        self.min_height = int(-80 / self.z_resolution) # -16
        self.agent_height = args.camera_height * 100. # default camera_height 0.88

        self.shift_loc = [self.vision_range * # [250, 0, pi/2]
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = get_camera_matrix(
            self.screen_w, self.screen_h, self.fov) # pixel: 160x120, field_of_view: 79

        self.vfov = np.arctan((self.screen_h/2.) / self.camera_matrix.f)
        self.min_vision = self.agent_height / np.tan(self.vfov) # cm

        self.pool = ChannelPool(1) # no usage. maxpooling of images' channel 

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, # default args.num_processes = 1
            self.vision_range, 
            self.vision_range, 
            self.max_height - self.min_height # torch.zeros(1,17,100,100,88)
        ).float().to(self.device)
        
        self.feat = torch.ones(
            args.num_processes, 
            1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device) # torch.ones(1,17,160*120)

    def export_point_cloud(self, points_depth, save_name):
        if points_depth.ndim > 3:
            point_cloud = points_depth.squeeze(0).cpu().numpy()
        points = point_cloud.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(f"./outputs/tmp/{save_name}.ply", pcd)
        print(f"saved at ./outputs/tmp/{save_name}.ply")

    def forward(self, obs, pose_obs, maps_last, poses_last, agent_heights):
        """
            obs: torch.Size([1, 20, 160, 120])
            pose_obs: torch.Size([1, 3]) | relative position from sensor's last position to current position
            maps_last: torch.Size([1, 20, 240, 240])
            poses_last: torch.Size([1, 3])
            agent_heights: sensor height, default = 0.88
        """

        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = get_point_cloud_from_z_t( # get point cloud from depth map
            depth, self.camera_matrix, self.device, scale=self.du_scale) # torch.Size([1, 160, 120, 3])
        # self.export_point_cloud(point_cloud_t, "point_cloud_t")
        agent_view_t = transform_camera_view_t( # transform pointc cloud according to sensor's height and elevation degree(仰角)
            point_cloud_t, agent_heights * 100., 0, self.device) # torch.Size([1, 160, 120, 3])
        # self.export_point_cloud(agent_view_t, "agent_view_t")
        agent_view_centered_t = transform_pose_t( # transform according to [x+250, y+0, rotate around z: pi/2]
            agent_view_t, self.shift_loc, self.device) # torch.Size([1, 160, 120, 3]) | just shift along x axis

        max_h = self.max_height # 72
        min_h = self.min_height # -16
        xy_resolution = self.resolution # 5
        z_resolution = self.z_resolution # 5
        vision_range = self.vision_range # 100

        
        XYZ_cm_std = agent_view_centered_t.float() # set dtype as float
        # self.export_point_cloud(XYZ_cm_std, "XYZ_cm_std")
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])
        
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)( # default du_scale = 1, then avgPool2d do nothing. If du_scale = 2, then avgPool2d do downscale of the image by averaging the 2x2 pixels
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale) # (1,16,160*120)

        voxels = splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        """
        Input:
            self.init_grid: (1,17,100,100,88) and set all elements as zero
            self.feat: (1,16,160*120)
            XYZ_cm_std: (1,3,160*120)
        Output:
            voxels: (1,17,100,100,88)
        """

        print("<<>><<>><<>> voxels", voxels.shape)

        min_z = int(35 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        floor_z = int(-35 / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)


        min_vision_std = int(self.min_vision // z_resolution)
        around_floor_proj = voxels[..., :min_z].sum(4) 
        under_floor_proj = voxels[..., floor_z:min_z].sum(4) 
        under_floor_proj = (under_floor_proj == 0.0).float()# have floor = 0, no floor or not deteced = 1
        under_floor_proj = under_floor_proj * around_floor_proj # no floor and detected = 1

        replace_element = torch.ones_like(depth[:, -1, :]) * self.min_vision
        re_depth = torch.where(depth[:, -1, :] < 3000, depth[:, -1, :], replace_element)
        count = ((re_depth - self.min_vision - 60) > 0).sum(dim=1)
        index = torch.nonzero(count > (re_depth.shape[1] / 4))

        under_floor_proj[index, 0:1, min_vision_std:min_vision_std+1, \
                (self.vision_range-6)//2 : (self.vision_range+6)//2] \
                    = 1.

        fp_map_pred = agent_height_proj[:, 0:1, :, :] + under_floor_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        """
        maps_last: torch.Size([1, 20, 240, 240])
        """

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses



class BEV_Map():
    def __init__(self, args):
        self.args = args

        self.num_scenes = self.args.num_processes
        nc = self.args.num_sem_categories + 4  # num channels
        self.device = self.args.device

        self.map_size = self.args.map_size
        self.global_width = self.args.global_width
        self.global_height = self.args.global_height
        self.local_width = self.args.local_width
        self.local_height = self.args.local_height

        self.mapping_module = Mapping(self.args).to(self.device)
        self.mapping_module.eval()

        # Initializing full and local map
        self.full_map = torch.zeros(self.num_scenes, nc, self.global_width, self.global_height).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, nc, self.local_width, self.local_height).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.local_map_boundary = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7)) # this is the variable communicate position computation
    
    def mapping(self, obs, infos):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 3: # 如果有3个维度，如(20,160,120)，则变为(1,20,160,120)
            obs = obs.unsqueeze(0)
        poses = torch.from_numpy(np.asarray(
            [infos['sensor_pose'] for env_idx
             in range(self.num_scenes)])
        ).float().to(self.device)
        agent_heights = torch.from_numpy(np.asarray(
            [infos['agent_height'] for env_idx in range(self.num_scenes)])
        ).float().to(self.device)

        _, self.local_map, _, self.local_pose = \
            self.mapping_module(obs, poses, self.local_map, self.local_pose, agent_heights) # SLAM | output: fp_map_pred, map_pred, pose_pred, current_poses
        
        # print("<<>><<>> obs:", obs.shape) # torch.Size([1, 20, 160, 120])
        # print("<<>><<>> poses:",  poses) # torch.Size([1, 3]) | relative position from sensor's last position to current position
        # print("<<>><<>> self.local_map:", self.local_map.shape) # torch.Size([1, 20, 240, 240])
        # print("<<>><<>> self.local_pose:", self.local_pose.shape) # torch.Size([1, 3])
        # print("<<>><<>> agent_heights:",  agent_heights) # agent_heights： sensor height, default = 0.88

        local_pose = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = local_pose + self.origins
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_scenes):
            # r, c = locs[e, 1], locs[e, 0]
            self.local_row = int(local_pose[e, 1] * 100.0 / self.args.map_resolution) # default map_resolution 5
            self.local_col = int(local_pose[e, 0] * 100.0 / self.args.map_resolution)
            self.local_map[e, 2:4, self.local_row - 2:self.local_row + 3, self.local_col - 2:self.local_col + 3] = 1.
    
    def move_local_map(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
        self.full_pose[env_idx] = self.local_pose[env_idx] + \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

        locs = self.full_pose[env_idx].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                        self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :,
                                self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1],
                                self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

    def get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_width // 2, loc_c - self.local_height // 2
            gx2, gy2 = gx1 + self.local_width, gy1 + self.local_height
            if gx1 < 0:
                gx1, gx2 = 0, self.local_width
            if gx2 > self.global_width:
                gx1, gx2 = self.global_width - self.local_width, self.global_width

            if gy1 < 0:
                gy1, gy2 = 0, self.local_height
            if gy2 > self.global_height:
                gy1, gy2 = self.global_height - self.local_height, self.global_height
        else:
            gx1, gx2, gy1, gy2 = 0, self.global_width, 0, self.global_height

        return [gx1, gx2, gy1, gy2]
    
    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.local_map_boundary[e] = self.get_local_map_boundaries((loc_r, loc_c))

            self.planner_pose_inputs[e, 3:] = self.local_map_boundary[e]
            self.origins[e] = [self.local_map_boundary[e][2] * self.args.map_resolution / 100.0,
                          self.local_map_boundary[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.local_map_boundary[e, 0]:self.local_map_boundary[e, 1],
                                    self.local_map_boundary[e, 2]:self.local_map_boundary[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

    def init_map_and_pose_for_env(self, env_idx=0):
        self.full_map[env_idx].fill_(0.)
        self.full_pose[env_idx].fill_(0.)
        self.full_pose[env_idx, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[env_idx].cpu().numpy()
        self.planner_pose_inputs[env_idx, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[env_idx, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                      self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

    def update_intrinsic_rew(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]