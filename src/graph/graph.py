import os
import sys
import cv2
import numpy as np
import torch
import math
import dataclasses
import omegaconf
from openai import OpenAI
import base64
from io import BytesIO
import supervision as sv
import random
from PIL import Image
from sklearn.cluster import DBSCAN  
from collections import Counter 
from omegaconf import DictConfig
from pathlib import PosixPath, Path
from supervision.draw.color import ColorPalette
import copy
import skimage

from .graphbuilder import GraphBuilder
from .goalgraphdecomposer import GoalGraphDecomposer
from .overlap import GraphMatcher
from .scenegraphcorrector import SceneGraphCorrector
from .utils.slam_classes import MapObjectList
from .utils.utils import filter_objects, gobs_to_detection_list
from .utils.mapping import compute_spatial_similarities, merge_detections_to_objects

from ..utils.fmm.fmm_planner import FMMPlanner
from ..utils.fmm import pose_utils as pu
from ..utils.camera import get_camera_matrix

sys.path.append('third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
from lightglue import LightGlue, DISK
from lightglue.utils import match_pair , numpy_image_to_torch

import clip


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."

class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SubGraph():
    def __init__(self, center_node):
        self.center_node = center_node
        self.edges = self.center_node.edges
        self.center = self.center_node.center
        self.nodes = set()
        for edge in self.edges:
            self.nodes.add(edge.node1)
            self.nodes.add(edge.node2)

    def get_subgraph_2_text(self):
        text = ''
        edges = set()
        for node in self.nodes:
            text = text + node.caption + '/'
            edges.update(node.edges)
        text = text[:-1] + '\n'
        for edge in edges:
            text = text + edge.relation + '/'
        text = text[:-1]
        return text


class Graph():
    def __init__(self, args, is_navigation=True) -> None:
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm
        self.map_size = args.map_size
        self.camera_matrix = get_camera_matrix(args.env_frame_height, args.env_frame_width, args.hfov)
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_width, full_height).int()
        self.device = args.device
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.is_navigation = is_navigation
        self.llm_name = args.llm_model
        self.vlm_name = args.vlm_model
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.set_cfg()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model = torch.jit.load("./third_party/clip/ViT-B-32.pt", map_location=self.device).eval()
        self.previous_mode = 1
        self.blacklist_node = []
        self.blacklist_edge = []
        self.previous_overlap = 0
        
        self.groundingdino_config_file = 'third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture'
        self.relations = ["next to", "opposite to", "below", "behind", "in front of"]
        # self.console = rich.console.Console()
        # self.prompt_llava = 'Describe the central object in this image.'
        self.prompt_llava = 'Use one word to describe the central object in this image.'
        # self.prompt_gpt = 'In a house, whether {} is often in a close proximity to a {}, please answer with "often" or "hardly".'
        self.prompt_gpt = '''
You need to predict the most likely distance of two objects in a room. You only need to answer "near" or "far". Unless the two are closely related, always answer "far". Here are two examples. 
1. table and chair, you should answer "near". (Because there is always a chair next to the table.)
2. door and cabinet, you should answer "far". (Because door and cabinets are not necessarily in the same place.)
Now predict the distance of {} and {}.
        '''
        self.prompt_gpt = '''
You need to predict the most likely distance of two objects in a room. You only need to answer the distance in meters. Here are 4 examples. 
1. book and bookshelf, you should answer "0.1". (Because book is in bookshelf.)
2. table and chair, you should answer "0.5". (Because there is always a chair next to the table.)
3. door and cabinet, you should answer "2". (Because door and cabinets are not necessarily in the same place but in the same room.)
4. bed and sofa, you should answer "5". (Because bed is in bedroom but sofa is in living room.)
Now predict the distance of {} and {}, and give your reason.
        '''
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character.
Examples:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to
Object pair(s):
        '''
        self.prompt_discriminate_relation = '''
In the image, do {} and {} satisfy the relationship of {}? Answer "yes" or "no".
        '''
        self.prompt_score_subgraph = '''
Please guess the most possible distance between these several objects and this target in the room respectively, in meters, ranging from 0.1 to 5 meters. Only answer one number.
This group of objects is {}.
The target is {}.
        '''
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_object_predict = 'The {} is most likely to appear near which of the following objects: {}'
        self.prompt_graph_corr = 'What is the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_image2text = 'Describe the object at the center of the image and indicate the spatial relationship between other objects and it.'
        self.prompt_create_relation = """
Given the image, please analyze the spatial relationship between {obj1} and {obj2}.
If there is a clear spatial relationship, describe it using the following template:
"{obj1} and {obj2}: {obj1} is <relation type> {obj2}"
If no clear spatial relationship exists, state: "No clear spatial relationship between {obj1} and {obj2}"

Example output format for a relation:
"table and book: table is under book"

Example output format if no relation:
"No clear spatial relationship between table and book"

Please provide the relationship you can determine from the image.
        """
        self.prompt_captions_filter = '''
Remove names that describe rooms of an indoor scene in the Given List. Only reply the processed list.
Examples:
Input: chair, windows, living room, cabinet, living room, kitchen, picture, tv
Output: chair, windows, cabinet, picture, tv
Given List: '''
        self.prompt_create_goal_edges = """
Given the image, please analyze the spatial relationship between {obj1} and {obj2}.
If there is a clear spatial relationship, describe it using the following template:
{obj1} and {obj2}: {obj1} is <relation type> {obj2}

Example output format for a relation:
table and book: table is under book

Please provide the relationship you can determine from the image.
        """
        self.grounded_sam = self.get_grounded_sam(self.device)
        # self.GSAM2_server_port = 7003
        # self.GSAM2_Client = GSAM2_Client(server_port=self.GSAM2_server_port)
        self.graphbuilder = GraphBuilder(self.get_llm_response)
        self.goalgraphdecomposer = GoalGraphDecomposer(self.get_llm_response)
        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.image_matcher = LightGlue(features='disk').eval().to(self.device)

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95,
               'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_obj_goal(self, obj_goal):
        self.obj_goal = obj_goal

    def set_image_goal(self, image):
    # goal graph construction like paper
        with torch.no_grad():
            mask, xyxy, masks_conf, caption = self.get_segmentation(self.grounded_sam, image)
        annotated_image = image.copy()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.instance_imagegoal = image

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=masks_conf,
            class_id=np.zeros_like(masks_conf).astype(int),
            mask=mask,
        )
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        # print("<><><><><><><><><> mask:",mask.shape)

        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = mask_annotator.annotate(annotated_image, detections)

        save_dir = "outputs/tmp"
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
        cv2.imwrite(os.path.join(save_dir, 'sii.jpg'), annotated_image[:, :, ::-1])
        caption_str = ', '.join(caption)
        prompt = f"{self.prompt_captions_filter} {caption_str}"
        filtered_caption = self.get_llm_response(prompt)
        print("<><><><><><> filtered_caption:", filtered_caption)
        # print("<><><><><><><><><> filtered_caption_length:", len(filtered_caption.split(', ')))
        filtered_objects = self.get_vlm_response(f"check all the objects in this list: {filtered_caption}, and output the objects that you can observe from the given image. Only reply the object list ", image)
        print("<><><><><><> filtered_objects:", filtered_objects)
        # print("<><><><><><><><><> filtered_objects_length:", len(filtered_objects.split(', ')))
        identified_objects_list = filtered_objects.split(', ')

        if self.obj_goal not in identified_objects_list:
            identified_objects_list.insert(0, self.obj_goal)
        print("<><><><><><> identified_objects_list:", identified_objects_list)
        
        edge_responses = ""
        for ia in range(len(identified_objects_list)-1):
            for ib in range(ia+1, len(identified_objects_list)):
                edge_response = self.get_vlm_response(self.prompt_create_goal_edges.format(obj1=identified_objects_list[ia], obj2=identified_objects_list[ib]), image)
                edge_responses = edge_responses+edge_response+"\n"
                
        if len(identified_objects_list) > 1:
            relations, objects = self.graphbuilder.get_relations_objects(edge_responses)
        elif len(identified_objects_list) == 1:
            relations = []
            objects = identified_objects_list
        else:
            relations = []
            objects = [self.obj_goal]
        self.goalgraph = self.graphbuilder.build_graph(objects, relations)
        print("<><><><><><><><><> New built Goal Graph:\n", self.goalgraph)
        self.goalgraph_decomposed = self.goalgraphdecomposer.goal_decomposition(self.goalgraph)
                

    # # default goal graph construction method
    #     if isinstance(image, np.ndarray):
    #         image = Image.fromarray(image)
    #     self.instance_imagegoal = image
    #     text_goal = self.get_vlm_response(self.prompt_image2text, self.instance_imagegoal)
    #     self.set_text_goal(text_goal)

    def set_text_goal(self, text_goal):
        self.text_goal = text_goal
        self.goalgraph = self.graphbuilder.build_graph_from_text(text_goal)
        self.goalgraph_decomposed = self.goalgraphdecomposer.goal_decomposition(self.goalgraph)

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_scenegraph(self):
        nodes = self.nodes
        edges = self.get_edges()
        caption_count = {}
        
        new_nodes = []
        
        node_id_map = {}
        for node in nodes:
            caption = node.caption
            if caption not in caption_count:
                caption_count[caption] = 0
            unique_id = "{}_{}".format(caption, caption_count[caption])
            new_nodes.append({
                'id': unique_id,
                'position': node.center
            })
            node_id_map[node] = unique_id
            caption_count[caption] += 1

        new_edges = []
        for edge in edges:
            source_id = node_id_map[edge.node1]
            target_id = node_id_map[edge.node2]
            new_edges.append({
                'source': source_id,
                'target': target_id,
                'type': edge.relation
            })

        self.scenegraph = {
            'nodes': new_nodes,
            'edges': new_edges
        }

        return self.scenegraph

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption
    
    def extract_node_and_edges(self, graph, target_node_id):
        # get nodes with the same id and take the first one as target_node
        target_node = next((node for node in graph['nodes'] if node['id'] == target_node_id), None)
        
        # 提取所有关联的边（包括作为source或target的边）
        connected_edges = [
            edge for edge in graph['edges'] 
            if edge['source'] == target_node_id or edge['target'] == target_node_id
        ]
        
        return {
            'nodes': [target_node],
            'edges': connected_edges
        }

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_grounded_sam(self, device):
        model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, device=device)
        predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
        return model, predictor
    
    def get_segmentation(
        self, model, image: np.ndarray
    ) -> tuple:
        groundingdino = model[0]
        sam_predictor = model[1]
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
    # here use grounded_sam
        boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
        if len(caption) == 0:
            return None, None, None, None
        sam_predictor.set_image(image)

        # size = image_pil.size
        H, W = image.shape[0], image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        mask, conf, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
        return mask, xyxy, conf, caption

    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        print('    segement2d...')
        with torch.no_grad():
            print('        sam_segmentation...')
            mask, xyxy, masks_conf, caption = self.get_segmentation(self.grounded_sam, self.image_rgb)

            self.seg_xyxy = xyxy
            self.seg_caption = caption
            self.clear_line()
        if caption is None:
            self.clear_line()
            return
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=masks_conf,
            class_id=np.zeros_like(masks_conf).astype(int),
            mask=mask,
        )
    # export segmentation pictures in stream
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_image = self.image_rgb.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = mask_annotator.annotate(annotated_image, detections)
        save_dir = "outputs/tmp"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 's.jpg'), annotated_image[:, :, ::-1])

        image_appear_efficiency = [''] * len(mask)
        self.segment2d_results.append({
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": self.classes,
            "image_appear_efficiency": image_appear_efficiency,
            "image_rgb": self.image_rgb,
            "caption": caption,
        })
        self.clear_line()
        # print("<><><><><><><><><> segmented caption:", caption)


    def mapping3d(self):
        print('    mapping3d...')
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]

        gobs = self.segment2d_results[-1]
        
        unt_pose = self.pose_matrix
        
        adjusted_pose = unt_pose
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = self.classes,
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation
        )
        
        if len(fg_detection_list) == 0:
            self.clear_line()
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            self.clear_line()
            return
                
        print('        compute_spatial_similarities...')
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        self.clear_line()
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')
        
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim)
        
        self.objects_post = filter_objects(self.cfg, self.objects)
        self.clear_line()
            
    def get_caption(self):
        print('    get_caption...')
        for idx, object in enumerate(self.objects_post):
            caption_list = []
            for idx_det in range(len(object["image_idx"])):
                caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                caption_list.append(caption)
            caption = self.find_modes(caption_list)[0]
            object['captions'] = [caption]
        self.clear_line()

    def update_node(self): # add new nodes to the scene graph
        print('    update_node...')
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            # caption_new = self.find_modes(self.objects_post[i]['captions'])[0]
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        # for i in range(node_num_ori, node_num_new):
        for new_object in new_objects:
            new_node = ObjectNode()
            # caption = self.find_modes(self.objects_post[i]['captions'])[0]
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            # self.create_new_edge(new_node)
            self.nodes.append(new_node)
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y
            node.set_center([x, y])
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
        self.clear_line()

    def create_new_edge(self, new_node):
        # new_edges = []
        for j, old_node in enumerate(self.nodes):
            image = self.get_joint_image(old_node, new_node)
            if image is not None:
                response = self.get_vlm_response(self.prompt_create_relation.format(obj1=old_node.caption, obj2=new_node.caption), image)
                if "No clear spatial relationship" not in response:
                    response = response.lower()
                    objects = [old_node.caption.lower(), new_node.caption.lower()]
                    relations = self.graphbuilder.get_relations(response, objects)
                    new_edge = Edge(old_node, new_node)
                    if len(relations) > 0:
                        new_edge.set_relation(relations[0]['type'])
                    else:
                        new_edge.set_relation('')
                    # new_node.edges.add(new_edge)
                    # old_node.edges.add(new_edge)
                    # new_edges.append(new_edge)

    def update_edge(self): # no edge prune like SG-Nav
        print('    update_edge...')
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            self.clear_line()
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                # new_node.edges.add(new_edge)
                # old_node.edges.add(new_edge)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                # new_node1.edges.add(new_edge)
                # new_node2.edges.add(new_edge)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            print(f'        LLM get all relation proposals...')
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.get_llm_response(prompt=prompt)
            relations = relations.split('\n')
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
            self.clear_line()
            # discriminate all relation proposals
            for i, new_edge in enumerate(new_edges):
                print(f'        discriminate_relation  {i}/{len(new_edges)}...')
                if new_edge.relation == None:
                    new_edge.delete()
                self.clear_line()
            # get edges set
            # self.edges = set()
            # for node in self.nodes:
            #     self.edges.update(node.edges)
        self.clear_line()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)
    
    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal
        print("<><><><><><> obj_goal:", self.obj_goal)
        print("<><><><><><> real_goal:", goal)
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        
        print("<><><><><><> room_node_text:", room_node_text)

        prompt = self.prompt_room_predict.format(goal, room_node_text) # goal graph of single goal with room nodes as prompt 
        response = self.get_llm_response(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center # group that most likely contrains goal in the predicted room as new distination to explore
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'update_observation {self.navigate_steps}...')
        self.segment2d()
        if len(self.segment2d_results) == 0:
            # self.clear_line()
            return
        self.mapping3d()
        self.get_caption()
        self.update_node()
        self.update_edge()
        self.get_scenegraph()
        # self.clear_line()

    def explore(self):
        # overlap = self.overlap()
        overlap = 0

        # if self.previous_mode == 2 and overlap < 0.9: 
        #     for nsi in self.matcher.common_pairs:
        #         if nsi["scene_idx"] not in self.blacklist_node:
        #             self.blacklist_node.append(nsi["scene_idx"])
        #     overlap = self.overlap()

        # print("<><><><><><> overlap <><><><><><>",overlap)
        if 0.5 <= overlap < 0.9 and len(self.matcher.common_nodes_goal) >= 2:
            goal = self.explore_remaining()
            self.previous_mode = 2
        elif overlap >= 0.9 and len(self.matcher.common_nodes_goal) < 2:
            goal = self.reasonableness_correction()
            self.previous_mode = 3
        else:
            goal = self.explore_subgraph() # set exploring the first subgraph as my goal
            self.previous_mode = 1
        
        # print("<><><><><><> blacklist_node <><><><><><>", self.blacklist_node)
        # print(f"<><><><><><> goal after mode {self.previous_mode} <><><><><><>", goal)
        goal = self.get_goal(goal)
        
        return goal

    def explore_subgraph(self, goal=None):
        if goal == None:
            self.subgraph = self.goalgraph_decomposed['subgraph_1']
        self.subgraph = self.goalgraphdecomposer.graph_to_text(self.subgraph) # here turns graph to text
        return self.insert_goal(self.subgraph)
    
    def overlap(self):
        graph1 = self.scenegraph
        graph2 = self.goalgraph
        black_list_node = self.blacklist_node
        black_list_edge = self.blacklist_edge
        # print("<><><> scene graph for overlap:",graph1)
        # print("<><><> goal graph for overlap:",graph2)

        scene_graph_nodes_embedded_features = torch.zeros(len(graph1['nodes']), 512)
        scene_graph_edges_embedded_features = torch.zeros(len(graph1['edges']), 512)
        goal_graph_nodes_embedded_features = torch.zeros(len(graph2['nodes']), 512)
        goal_graph_edges_embedded_features = torch.zeros(len(graph2['edges']), 512)
        with torch.no_grad():
            for idx, nodei in enumerate(graph1['nodes']):
                node_text_feature = nodei['id'].split('_')[0]
                text_token = clip.tokenize([node_text_feature]).to(self.device)
                embedded_feature = self.clip_model.encode_text(text_token)
                scene_graph_nodes_embedded_features[idx] = embedded_feature

            for idx, edgei in enumerate(graph1['edges']):
                edges_text_feature = f"{edgei['source']} {edgei['type'].strip()} {edgei['target']}"
                text_token = clip.tokenize([edges_text_feature]).to(self.device)
                embedded_feature = self.clip_model.encode_text(text_token)
                scene_graph_edges_embedded_features[idx] = embedded_feature

            for idx, nodei in enumerate(graph2['nodes']):
                node_text_feature = nodei['id']
                text_token = clip.tokenize([node_text_feature]).to(self.device)
                embedded_feature = self.clip_model.encode_text(text_token)
                goal_graph_nodes_embedded_features[idx] = embedded_feature

            for idx, edgei in enumerate(graph2['edges']):
                edges_text_feature = f"{edgei['source']} {edgei['type'].strip()} {edgei['target']}"
                text_token = clip.tokenize([edges_text_feature]).to(self.device)
                embedded_feature = self.clip_model.encode_text(text_token)
                goal_graph_edges_embedded_features[idx] = embedded_feature
            
        
        self.matcher = GraphMatcher(graph1, graph2, self.get_llm_response)
        overlap_score = self.matcher.overlap(
            goal_node_features = goal_graph_nodes_embedded_features,
            goal_edge_features = goal_graph_edges_embedded_features,
            scene_node_features = scene_graph_nodes_embedded_features, 
            scene_edge_features = scene_graph_edges_embedded_features,
            black_list_node = black_list_node,
            black_list_edge = black_list_edge,
        )
        return overlap_score
    
    def explore_remaining(self):
        G1 = self.matcher.G1
        G2 = self.matcher.G2
        common_pairs = self.matcher.common_pairs
        common_nodes_goal = set(self.matcher.common_nodes_goal)

        # Assign positions to the first two common nodes in the subgraph
        for i, pairi in enumerate(common_pairs):
            if i < 2:
                G2.nodes[list(G2.nodes)[pairi['goal_idx']]]['position'] = G1.nodes[list(G1.nodes)[pairi['scene_idx']]]['position']
            else:
                break

        # Calculate relative positions within the subgraph
        positions = self.matcher.calculate_relative_positions(G2, common_nodes_goal)

        # print("><><><><><>< positions", positions)

        # Predict positions of the remaining nodes
        position = self.matcher.predict_remaining_node_positions(common_pairs, positions, G1)
        return position

    def reasonableness_correction(self):
        corrector = SceneGraphCorrector(self.get_llm_response)
        self.scenegraph = corrector.correct_scene_graph(self.scenegraph, self.obj_goal)
        return None

    def clear_line(self, line_num=1): # clear the line in shell
        for i in range(line_num):  
            sys.stdout.write('\033[F')
            sys.stdout.write('\033[J')
            sys.stdout.flush()  

    def get_llm_response(self, prompt):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # print(">>>>>>>>>LLM_sent_message>>>>>>>>>\n", prompt)
        # print(">>>>>>>>>LLM_sent_message>>>>>>>>>")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.llm_name,
        )
        # print("<<<<<<<<<LLM_reponse<<<<<<<<<\n", chat_completion.choices[0].message.content)
        # print("<<<<<<<<<LLM_reponse<<<<<<<<<")
        return chat_completion.choices[0].message.content
    
    def get_vlm_response(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')

        # print(">>>>>>>>>VLM_sent_message>>>>>>>>>\n", prompt)
        # print(">>>>>>>>>VLM_sent_message>>>>>>>>>")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_str}}
                    ]
                }
            ],
            model=self.llm_name,
        )
        # print("<<<<<<<<<VLM_reponse<<<<<<<<<\n", chat_completion.choices[0].message.content)
        # print("<<<<<<<<<VLM_reponse<<<<<<<<<")
        return chat_completion.choices[0].message.content
    
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def get_goal(self, goal=None):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        if self.full_map.shape[1] == 1:
            fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        else:
            fbe_map[self.full_map[0,1]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_map = frontier_map & (self.num_of_goal < 3).to(frontier_map.device)
        # frontier_map = self.frontier_map
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, start = self.get_traversible(self.full_map.cpu().numpy()[0, 0, ::-1], input_pose)
        planner = FMMPlanner(traversible)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        distance_threshold = 3
        idx_16 = np.where(distances>=distance_threshold)
        distances_16 = distances[idx_16]
        distances_16_inverse = 10 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175
        scores = np.zeros((num_16_frontiers))
        
        scores += distances_16_inverse
        if isinstance(goal, list) or isinstance(goal, np.ndarray):
            goal = list(goal)

            planner = FMMPlanner(traversible)
            state = [goal[0] + 1, goal[1] + 1]
            planner.set_goal(state)
            fmm_dist = planner.fmm_dist[::-1]
            frontier_locations += 1
            if isinstance(frontier_locations, torch.Tensor):
                frontier_locations = frontier_locations.cpu().numpy()
            distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
            
            ## use the threshold of 1.6 to filter close frontiers to encourage exploration
            distance_threshold = 3
            idx_16 = np.where(distances>=distance_threshold)
            distances_16 = distances[idx_16]
            distances_16_inverse = 1 - (np.clip(distances_16, 0, 10 + distance_threshold) - distance_threshold) / 10
            frontier_locations_16 = frontier_locations[idx_16]
            self.frontier_locations = frontier_locations
            self.frontier_locations_16 = frontier_locations_16
            if len(distances_16) == 0:
                return None
            num_16_frontiers = len(idx_16[0])
            scores = np.zeros((num_16_frontiers))
            scores += distances_16_inverse

        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        return goal
    
    def get_traversible(self, map_pred, pose_pred):
        if isinstance(map_pred, torch.Tensor):
            map_pred = map_pred.cpu().numpy()
        if len(map_pred.shape) == 4:
            map_pred = map_pred[0, 0]
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        selem = skimage.morphology.square(1)
        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start
    
    def reset(self):
        full_width, full_height = self.map_size, self.map_size
        self.full_width = full_width
        self.full_height = full_height
        self.visited = torch.zeros(full_width, full_height).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_width, full_height).int()
        self.segment2d_results = []
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_list = []
        self.blacklist_node = []
        self.blacklist_edge = []
        self.get_scenegraph()

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.get_llm_response(prompt=prompt)
        corr_score = self.text2value(response_3)
        return corr_score
    
    def text2value(self, text):
        try:
            value = float(text)
        except:
            value = 0
        return value
    