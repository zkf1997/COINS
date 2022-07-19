import os

import numpy as np

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
sys.path.append('..')
# sys.path.append('../POSA')
# sys.path.append('../human_body_prior/src')

import smplx
import open3d as o3d
import trimesh
from datetime import datetime
import pickle
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch3d
from pytorch3d.structures import Pointclouds, Meshes
import pytorch3d.loss
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
import open3d as o3d
from datetime import datetime
from copy import deepcopy
from argparse import ArgumentParser

from configuration.config import *
from data.scene import scenes, to_trimesh, to_open3d
from data.utils import *
from interaction.chamfer_distance import chamfer_contact_loss, chamfer_dists
from interaction.mesh import Mesh
import pyrender

def transform_back(vertices, centroid, rotation):
    B, N, C = vertices.shape
    vertices = vertices.matmul(torch.inverse(rotation).transpose(1, 2))
    vertices = vertices + centroid.unsqueeze(1)
    return vertices

def get_contact_vertices(verb_codes, body_part_vertices):
    contact_vertices = []
    batch_size = verb_codes.shape[0]
    for batch_idx in range(batch_size):
        verb_code = verb_codes[batch_idx]
        # print(verb_code)
        body_parts = set()
        for idx, num in enumerate(verb_code):
            if num == 1:
                # print(action_names[idx], action_body_part_mapping[action_names[idx]])
                body_parts.update(action_body_part_mapping[action_names[idx]])
        # print(body_parts)
        vertices_idx = []
        for body_part in body_parts:
            vertices_idx += body_part_vertices[body_part]
        contact_vertices.append(vertices_idx)
    return contact_vertices

# chamferDist = ChamferDistance()
# def calc_contact_loss(body_vertices, contact_vertex_idx, obj_points):
#     batch_size = body_vertices.shape[0]
#     contact_vertices = []
#     for batch_idx in range(batch_size):
#         contact_vertices.append(body_vertices[batch_idx, contact_vertex_idx[batch_idx], :])
#     loss, _ = chamfer_contact_loss(Pointclouds(contact_vertices), obj_points[:, :, :3], batch_reduction='mean')
#     # not use Geman-McClure error function since we use single object,
#     # loss_contact = (fcc * self.weight_contact *
#     #                 torch.mean(torch.sqrt(contact_dist + 1e-4)
#     #                            / (torch.sqrt(contact_dist + 1e-4) + 1.0)))
#     return loss

def calc_contact_loss(body_vertices, verb_ids, contact_vertices_list, obj_points):
    batch_size = body_vertices.shape[0]
    B, I, P, C = obj_points.shape
    contact_vertices = []
    loss_list = []
    verb_ids = verb_ids.clone().long()
    for batch_idx in range(batch_size):
        verb_id = verb_ids[batch_idx]
        if (verb_id[1] == -1):
            verb_id[1] = verb_id[0]
        # print(verb_id)
        contact_vertices.append(body_vertices[batch_idx, contact_vertices_list[verb_id[0]], :])
        contact_vertices.append(body_vertices[batch_idx, contact_vertices_list[verb_id[1]], :])

    dists = chamfer_dists(Pointclouds(contact_vertices), obj_points[:, :, :, :3].reshape(B*I, P, 3)).reshape(B, I, -1)
    # print(dists.shape)

    for batch_idx in range(batch_size):
        verb_id = verb_ids[batch_idx]
        if (verb_id[1] == -1):
            verb_id[1] = verb_id[0]
        for atomic_idx in range(maximum_atomics):
            num_vertices = len(contact_vertices_list[verb_id[atomic_idx]])
            loss = torch.mean(dists[batch_idx, atomic_idx, :num_vertices]) if \
            verb_id[atomic_idx] != 3 else torch.min(
                dists[batch_idx, atomic_idx, :num_vertices // 2].mean(),
                dists[batch_idx, atomic_idx, num_vertices // 2:].mean()
            )  # for touch, use the minimum of mean contact dists of two hands sicne can touch with only one hand, for other three, all realted parts shall have close contact
            loss_list.append(loss)

    # loss, _ = chamfer_contact_loss(Pointclouds(contact_vertices), obj_points[:, :, :, :3].reshape(B * I, P, 3),
    #                                batch_reduction='mean', point_reduction='mean')

    return torch.stack(loss_list).mean()

def get_scene_sdfs(scene_names, device):
    sdf_grids = []
    sdf_max = []
    sdf_min = []
    for scene_name in scene_names:
        if not hasattr(scenes[scene_name], 'sdf_torch'):
            scenes[scene_name].sdf_torch = torch.from_numpy(scenes[scene_name].sdf).squeeze().unsqueeze(0).unsqueeze(0).to(device) # 1x1xDxDxD
        sdf_grids.append(scenes[scene_name].sdf_torch)
        # sdf_grids.append(torch.from_numpy(scenes[scene_name].sdf).squeeze().unsqueeze(0).unsqueeze(0).to(device)) # 1x1xDxDxD
        sdf_config = scenes[scene_name].sdf_config
        sdf_max.append(torch.tensor(sdf_config['grid_max']).reshape(1, 1, 3))
        sdf_min.append(torch.tensor(sdf_config['grid_min']).reshape(1, 1, 3))
    sdf_grids = torch.cat(sdf_grids, dim=0)
    sdf_max = torch.cat(sdf_max, dim=0).to(sdf_grids.device)
    sdf_min = torch.cat(sdf_min, dim=0).to(sdf_grids.device)
    return sdf_grids, sdf_min, sdf_max

def calc_penetration_loss(scene_sdfs, body_vertices, thresh=0.0):
    sdf_grids, sdf_min, sdf_max = scene_sdfs
    batch_size, num_vertices, _ = body_vertices.shape
    body_vertices = ((body_vertices - sdf_min)
                                / (sdf_max - sdf_min) *2 -1)
    body_sdf_batch = F.grid_sample(sdf_grids,
                                   body_vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3),
                                   padding_mode='border',
                                   align_corners=True  # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   )
    # if there are no penetrating vertices then set sdf_penetration_loss = 0

    if body_sdf_batch.lt(thresh).sum().item() < 1:
        loss_sdf_pene = torch.tensor(0.0, dtype=torch.float32,
                                     device=body_vertices.device)
    else:
        body_sdf_batch = body_sdf_batch - thresh
        loss_sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()
    return loss_sdf_pene


def eval_and_vis():
    loss_contact = calc_contact_loss(body_vertices, contact_vertices, obj_points)
    print(loss_contact)
    scene_sdfs = get_scene_sdfs([scene_name])
    loss_penetration = calc_penetration_loss(scene_sdfs, body_vertices)
    print(loss_penetration)

    colors = np.ones((body_vertices.shape[1], 3)) * np.array([0.8, 0.8, 0.8])
    colors[contact_vertices[0], :] = np.ones((len(contact_vertices[0]), 3)) * np.array([0.1, 0.8, 0.1])
    body = trimesh.Trimesh(vertices=body_vertices[0].detach().cpu().numpy(),
                           faces=mesh.faces,
                           vertex_colors=colors)
    obj_mesh = scenes[scene_name].get_mesh_with_accessory(node_idx)
    obj_pointcloud = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(obj_points[0].detach().cpu().numpy())
    )
    obj_pointcloud.paint_uniform_color((0.8, 0.1, 0.1))
    geometries = [to_open3d(body), to_open3d(obj_mesh), obj_pointcloud
                  ]
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    # data
    with open(Path.joinpath(project_folder, "data", 'train.pkl'), 'rb') as data_file:
        train_data = pickle.load(data_file)
    with open(Path.joinpath(project_folder, "data", 'test.pkl'), 'rb') as data_file:
        test_data = pickle.load(data_file)

    used_interaction = 'sit on-sofa'
    num_points = 4096
    train_dataset = SingleObjectDataset(train_data, num_points=num_points, use_augment=True,
                                        used_interaction='sit on-sofa'
                                        )
    # test_dataset = SingleObjectDataset(test_data, num_points=num_points, use_augment=False,
    #                                    used_interaction=used_interaction
    #                                    )
    device = torch.device('cuda')
    mesh = Mesh(num_downsampling=2)
    for frame in range(3000, train_dataset.__len__(), 30):
        smplx_param, pelvis, body_vertices, obj_points, centroid, scale, rotation, \
        atomic_interaction, obj_category_code, verb_code, scene_name, node_idx = train_dataset.__getitem__(frame)
        body_vertices = torch.tensor(body_vertices + pelvis.reshape((1, 3)), device=device).unsqueeze(0)
        body_vertices = mesh.downsample(body_vertices)
        obj_points = torch.tensor(obj_points[:, :3], device=device).unsqueeze(0)
        verb_code = torch.tensor(verb_code, device=device).unsqueeze(0)
        centroid = torch.tensor(centroid, device=device).unsqueeze(0)
        rotation = torch.tensor(rotation, device=device).unsqueeze(0)
        contact_vertices = get_contact_vertices(verb_code, mesh.body_part_vertices)
        body_vertices = transform_back(body_vertices, centroid, rotation)
        print(scale)
        obj_points = transform_back(obj_points * scale, centroid, rotation)

        eval_and_vis()
        body_vertices = body_vertices - torch.tensor([0.0, 0.0, .2], dtype=torch.float32, device=device)
        eval_and_vis()


