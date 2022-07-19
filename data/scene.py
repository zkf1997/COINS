import os
import sys
import time

sys.path.append('..')
from configuration.config import *

import copy
from copy import deepcopy
import json
import os
import trimesh
import pickle
import torch
from collections import defaultdict
from scipy.spatial import KDTree

import numpy as np
import torch.nn.functional as F
import open3d as o3d
from sklearn.decomposition import PCA

DEBUG = False

def to_trimesh(o3d_mesh):
    # if isinstance(o3d_mesh, trimesh.Trimesh):
    #     return o3d_mesh
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles),
        vertex_colors=np.asarray(o3d_mesh.vertex_colors),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals), process=False
    )

def to_open3d(trimesh_mesh):
    trimesh_mesh = deepcopy(trimesh_mesh)  # if not copy, vertex normal cannot be assigned
    o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                         triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))
    # as_open3d method not working for color and normal
    if hasattr(trimesh_mesh.visual, 'vertex_colors'):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(trimesh_mesh.visual.vertex_colors[:, :3] / 255.0)
    o3d_mesh.compute_vertex_normals()  # if not compute but only assign trimesh normals, the normal rendering fails, not sure about the reason
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)
    # input_normals = trimesh_mesh.vertex_normals
    # print('normal_diff:', input_normals - np.asarray(o3d_mesh.vertex_normals))
    return o3d_mesh

def select_face_by_vertices(faces, vertices):
    return [face for face in faces
            if set(face).issubset(set(vertices))]

def get_obb_extents(mesh):
    return trimesh.bounds.oriented_bounds(mesh)[1]

def bbox_intersect(aabb1, aabb2):
    min_bound = np.maximum(np.asarray(aabb1.min_bound), np.asarray(aabb2.min_bound))
    max_bound = np.minimum(np.asarray(aabb1.max_bound), np.asarray(aabb2.max_bound))
    return (min_bound[:2] < max_bound[:2]).any()

def get_intersection_2d(aabb1, aabb2):
    min_bound = np.maximum(aabb1[:2], aabb2[:2])
    max_bound = np.minimum(aabb1[2:], aabb2[2:])
    return np.concatenate([min_bound, max_bound])

class ObjectNode:
    def __init__(self, load=False, **kwargs):
        # load from serialization
        if load:
            self.__dict__.update(kwargs)
        else:
            self.id = kwargs.get('id')
            self.scene = kwargs.get('scene')
            self.category = kwargs.get('category')
            self.category_name = kwargs.get('category_name') if 'category_name' in kwargs else category_dict.loc[self.category]['mpcat40']
            self.vis_color = kwargs.get('vis_color')
            self.mesh = kwargs.get('mesh')
            self.aabb = kwargs.get('aabb')
            self.pointcloud = kwargs.get('pointcloud')
            # self.trans = normalize_transformation(self.pointcloud)

            # calculate object node features if not given from serialization
            # pigraph segment features
            self.hc = None  # centroid height above ground
            self.hs = None  # oriented bounding box height extent
            self.dxy = None  # diagoanl length of xy plane of obb
            self.axy = None  # area of xy plane of obb
            self.dominant_normal = None  # dominant normal (min PCA axis)
            self.z = None  # dot product of dominant normal(min PCA axis) with upwards vector (0,0,1), pigraph seems to use the mean of per point normals as dominant normal in code
            self.features = None  # concatenate features above into a vector
            self.calc_features()
        vis_mesh = o3d.geometry.TriangleMesh(self.mesh)
        vis_mesh.paint_uniform_color(self.vis_color)
        vis_mesh.vertex_normals = self.mesh.vertex_normals
        self.vis_mesh = vis_mesh

    def calc_features(self):
        centroid = np.asarray(self.pointcloud.points).mean(axis=0)
        obb = self.pointcloud.get_oriented_bounding_box()
        covariance = self.pointcloud.compute_mean_and_covariance()[1]
        eigen_values, eigen_vectors = np.linalg.eig(covariance)
        # pigraph segment features
        self.hc = centroid[2]   # centroid height above ground
        self.hs = obb.extent[2]  # oriented bounding box height extent
        self.dxy = np.sqrt(obb.extent[0] * obb.extent[0] + obb.extent[1] * obb.extent[1])  # diagoanl length of xy plane of obb
        self.axy = obb.extent[0] * obb.extent[1]  # area of xy plane of obb
        self.dominant_normal = eigen_vectors[-1]  # dominant normal (min PCA axis)
        self.z = eigen_vectors[-1][2] # dot product of dominant normal(min PCA axis) with upwards vector (0,0,1), pigraph seems to use the mean of per point normals as dominant normal in code
        self.features = [self.hc, self.hs, self.dxy, self.axy, self.z]

    def serialize(self):
        serialized = copy.deepcopy(self.__dict__)
        serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
                              'triangles': np.asarray(self.mesh.triangles),
                              'vertex_colors': np.asarray(self.mesh.vertex_colors),
                              'vertex_normals': np.asarray(self.mesh.vertex_normals)}
        del serialized['aabb']
        del serialized['pointcloud']
        del serialized['vis_mesh']
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
        vertex_normals = o3d.utility.Vector3dVector(serialized['mesh']['vertex_normals'])
        serialized['mesh'] = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
            triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
        )
        serialized['mesh'].vertex_colors = vertex_colors
        serialized['mesh'].compute_vertex_normals()
        serialized['mesh'].vertex_normals = vertex_normals
        serialized['aabb'] = serialized['mesh'].get_axis_aligned_bounding_box()
        serialized['pointcloud'] = o3d.geometry.PointCloud(serialized['mesh'].vertices)
        return cls(load=True, **serialized)

class Scene:
    def __init__(self, scene_name):
        # load from serialization
        scene_cache_path = os.path.join(scene_cache_folder, scene_name + '.pkl')
        assert os.path.exists(scene_cache_path)
        with open(scene_cache_path, 'rb') as f:
            try:
                serialized = pickle.load(f)
                vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
                vertex_normals = o3d.utility.Vector3dVector(serialized['mesh']['vertex_normals'])
                serialized['mesh'] = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
                    triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
                )
                serialized['mesh'].vertex_colors = vertex_colors
                serialized['mesh'].compute_vertex_normals()
                serialized['mesh'].vertex_normals = vertex_normals

                serialized['object_nodes'] = [ObjectNode.deserialize(obj_node) for obj_node in
                                              serialized['object_nodes']]
                self.__dict__.update(**serialized)
            except Exception as e:
                print(e, "load failed")

        # build kdtree
        # points = np.zeros((0, 3))
        # self.vertex_number_list = [] # vertex number of each object, used to query which object a point belong to
        # for obj_node in self.object_nodes:
        #     self.vertex_number_list.append(np.asarray(obj_node.mesh.vertices).shape[0])
        #     points = np.concatenate((points, np.asarray(obj_node.mesh.vertices)), axis=0)
        # self.vertex_number_list = np.asarray(self.vertex_number_list)
        # self.vertex_number_sum = np.cumsum(self.vertex_number_list)
        # self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # self.kdtree = o3d.geometry.KDTreeFlann(self.pointcloud) # kdtree of all object instances

        # load sdf
        # https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        with open(os.path.join(sdf_folder, self.name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_dim = sdf_data['dim']
            grid_min = np.array(sdf_data['min']).astype(np.float32)
            grid_max = np.array(sdf_data['max']).astype(np.float32)
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(os.path.join(sdf_folder, self.name + '_sdf.npy')).astype(np.float32)
        sdf = sdf.reshape((grid_dim, grid_dim, grid_dim, 1))
        self.sdf = sdf
        # self.sdf_torch = torch.from_numpy(sdf.reshape((1, grid_dim, grid_dim, grid_dim))).to(
        #     torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.sdf_config = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim}
        self.mesh_with_accessory = {}

    # def vertex_id_to_obj_id(self, vertex_id):
    #     return np.searchsorted(self.vertex_number_sum, vertex_id)

    # def serialize(self):
    #     serialized = copy.deepcopy(self.__dict__)
    #     serialized['mesh'] = {'vertices': np.asarray(self.mesh.vertices),
    #                           'triangles': np.asarray(self.mesh.triangles),
    #                           'vertex_colors': np.asarray(self.mesh.vertex_colors),
    #                           'vertex_normals': np.asarray(self.mesh.vertex_normals)}
    #     serialized['object_nodes'] = [obj_node.serialize() for obj_node in serialized['object_nodes']]
    #     # del serialized['kdtree']
    #     return serialized
    #
    # @classmethod
    # def deserialize(cls, serialized):
    #     # print(serialized['mesh']['vertices'].shape)
    #     vertex_colors = o3d.utility.Vector3dVector(serialized['mesh']['vertex_colors'])
    #     vertex_normals = o3d.utility.Vector3dVector(serialized['mesh']['vertex_normals'])
    #     serialized['mesh'] = o3d.geometry.TriangleMesh(
    #         vertices=o3d.utility.Vector3dVector(serialized['mesh']['vertices']),
    #         triangles=o3d.utility.Vector3iVector(serialized['mesh']['triangles'])
    #     )
    #     serialized['mesh'].vertex_colors = vertex_colors
    #     serialized['mesh'].compute_vertex_normals()
    #     serialized['mesh'].vertex_normals = vertex_normals
    #
    #     serialized['object_nodes'] = [ObjectNode.deserialize(obj_node) for obj_node in serialized['object_nodes']]
    #     # points = np.zeros((0, 3))
    #     # for obj_node in serialized['object_nodes']:
    #     #     points = np.concatenate((points, np.asarray(obj_node.mesh.vertices)), axis=0)
    #     # serialized['kdtree'] = o3d.geometry.KDTreeFlann(points)
    #     return cls(load=True, **serialized)
    #
    # @classmethod
    # # create from cache if available or compute using scene scan
    # def create(cls, scene_name, overwrite=False):
    #     # check if there are precomputed scene nodes
    #     scene_cache_path = os.path.join(scene_cache_folder, scene_name + '.pkl')
    #     if os.path.exists(scene_cache_path) and not overwrite:
    #         with open(scene_cache_path, 'rb') as f:
    #             try:
    #                 pkl = pickle.load(f)
    #                 print("create with cache for:", scene_name)
    #                 return cls.deserialize(pkl)
    #             except Exception as e:
    #                 print(e, "load failed, try to build scene nodes from scene scan")
    #                 return cls(scene_name=scene_name)
    #     print("build scene nodes from scan for scene:", scene_name)
    #     return cls(scene_name=scene_name)

    def get_visualize_geometries(self, semantic=False):
        # if not semantic:
        #     return [self.mesh]
        geometries = []
        # visualize scene
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        geometries.append(mesh_frame)
        if self.name in ['Werkraum', 'MPH1Library'] and not semantic:
            geometries.append(self.mesh)
        else:
            for object_node in self.object_nodes:
                # if object_node.category_name != 'table':
                #     continue
                geometries.append(object_node.vis_mesh if semantic else object_node.mesh)
                geometries.append(object_node.aabb)
        return geometries

    def visualize(self, trans=None, semantic=False):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 20.0

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        vis.poll_events()
        vis.update_renderer()
        vis.run()

    def save(self, semantic=True):
        # render
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, top=0, left=0, visible=False)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().line_width = 50

        geometries = self.get_visualize_geometries(semantic=semantic)
        for geometry in geometries:
            vis.add_geometry(geometry)

        if self.cam2world is not None:
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.extrinsic = np.linalg.inv(self.cam2world)
            ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        vis_type = 'semantic' if semantic else 'raw'
        filename = os.path.join(scene_cache_folder, self.name + '_' + vis_type + '.png')
        vis.capture_screen_image(filename, True)
        vis.destroy_window()

    def log(self):
        file_path = os.path.join(scene_cache_folder, self.name + '.txt')
        with open(file_path, 'w') as f:
            idx = 0
            for object_node in self.object_nodes:
                print(("{idx}: {obj} of {num} vertices\n".format(idx=idx, obj=object_node.category_name + '_' + str(object_node.id),
                                                      num=np.asarray(object_node.mesh.vertices).shape[0]
                                                      )))
                f.write("{idx}: {obj} of {num} vertices\n".format(idx=idx, obj=object_node.category_name + '_' + str(object_node.id),
                                                      num=np.asarray(object_node.mesh.vertices).shape[0]
                                                      ))
                idx += 1

    # return complete meshes for single instances of sofa, bed, table by adding back accessory objects like cushion and objects
    def get_mesh_with_accessory(self, node_idx):
        if node_idx in self.mesh_with_accessory:
            return self.mesh_with_accessory[node_idx]

        # obj = [obj for obj in self.object_nodes if obj.id == node_idx][0]  # for scannet, index in the list can be different from instance index because 0 is skipped
        obj = self.object_nodes[node_idx]
        mesh = to_trimesh(obj.mesh)
        if obj.category_name in ['sofa', 'bed', 'table', 'cabinet', 'chest_of_drawers']:
            accessory_list = ['objects', 'object']
            if obj.category_name == 'table':
                accessory_list += ['tv_monitor']
            if obj.category_name in ['sofa', 'bed', 'couch']:
                accessory_list += ['cushion', 'pillow', 'pillows']
            accessory_candidates = [obj for obj in self.object_nodes if obj.category_name in accessory_list]
            if len(accessory_candidates):
                proximity = KDTree(mesh.vertices)
                for candidate in accessory_candidates:
                    dists, _ = proximity.query(np.asarray(candidate.mesh.vertices))
                    if dists.min() < 0.1:
                        mesh += to_trimesh(candidate.mesh)
        self.mesh_with_accessory[node_idx] = mesh
        return mesh

    def support_interaction(self, query_interaction):
        atomic_interactions = query_interaction.split('+')
        nouns = [atomic.split('-')[1] for atomic in atomic_interactions]
        scene_objs = [node.category_name for node in self.object_nodes]
        return set(nouns).issubset(set(scene_objs))

    def get_interaction_candidate_objects(self, interaction, use_annotation=True):
        atomic_interactions = interaction.split('+')
        verbs = [atomic.split('-')[0] for atomic in atomic_interactions]
        nouns = [atomic.split('-')[1] for atomic in atomic_interactions]
        if use_annotation and self.name in candidate_combination_dict and '+'.join(nouns) in candidate_combination_dict[self.name]:
            # print('use annotation')
            candidate_combination = candidate_combination_dict[self.name]['+'.join(nouns)]
            candidate_combination = [[self.object_nodes[obj_idx] for obj_idx in obj_combination] for obj_combination in candidate_combination]
            return verbs, nouns, candidate_combination
        else:
            # print('automic filtering', interaction)
            candidate_combination = None
            for noun in nouns:
                candidate_instances = [node for node in self.object_nodes if node.category_name == noun]
                if candidate_combination is None:
                    candidate_combination = [[instance] for instance in candidate_instances]
                else:
                    updated_combination = []
                    for combination in candidate_combination:
                        for instance in candidate_instances:
                            updated_combination.append(combination + [instance])
                    candidate_combination = updated_combination
        return verbs, nouns, candidate_combination

    def get_floor_height(self):
        if hasattr(self, 'floor_height'):
            return  self.floor_height
        floor_pointclouds = [np.asarray(obj_node.mesh.vertices) for obj_node in self.object_nodes if obj_node.category_name == 'floor']
        if len(floor_pointclouds) == 0:
            floor_height = self.mesh.get_min_bound()[2]
        else:
            max_idx = np.argmax(np.array(
                [pointcloud.shape[0] for pointcloud in floor_pointclouds]
            ))
            max_floor = floor_pointclouds[max_idx]
            floor_height = max_floor[:, 2].mean()  # mean of z coord of points of max sized floor
        self.floor_height = floor_height
        return floor_height

    def calc_sdf(self, vertices):
        if not hasattr(self, 'sdf_torch'):
            self.sdf_torch = torch.from_numpy(self.sdf).squeeze().unsqueeze(0).unsqueeze(0) # 1x1xDxDxD
        sdf_grids = self.sdf_torch.to(vertices.device)
        sdf_config = self.sdf_config
        sdf_max = torch.tensor(sdf_config['grid_max']).reshape(1, 1, 3).to(vertices.device)
        sdf_min = torch.tensor(sdf_config['grid_min']).reshape(1, 1, 3).to(vertices.device)

        # vertices = torch.tensor(vertices).reshape(1, -1, 3)
        batch_size, num_vertices, _ = vertices.shape
        vertices = ((vertices - sdf_min)
                         / (sdf_max - sdf_min) * 2 - 1)
        sdf_values = F.grid_sample(sdf_grids,
                                       vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                       padding_mode='border',
                                       align_corners=True
                                       )
        return sdf_values.reshape(batch_size, num_vertices)

scenes = {}
for scene_name in scene_names:
    scenes[scene_name] = Scene(scene_name=scene_name)
if __name__ == "__main__":
    for scene_name in scene_names:
        scene = scenes[scene_name]
        scene.log()
        scene.save(semantic=True)
        scene.save(semantic=False)
        print('floor height:', scene.get_floor_height())




