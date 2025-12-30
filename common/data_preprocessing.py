import numpy as np
import torch
from abc import ABC, abstractmethod
from constants import AIRCRAFT_COUNT
from torch import Tensor
from torch_geometric.data import Data
from typing import Tuple


AC_FAMILY_MAPPING = {
    "B737": "B737NG",
    "B738": "B737NG",
    "B739": "B737NG",
    "A359": "A350",
    "A35K": "A350",
    "B752": "B757",
    "B772": "B777",
    "B773": "B777",
    "B77W": "B777",
    "B77L": "B777",
    "B788": "B787",
    "B789": "B787",
    "B78X": "B787",
    "A319": "A320",
    "A320": "A320",
    "A321": "A320",
    "A21N": "A320neo",
    "A20N": "A320neo",
    "B744": "B747",
    "B748": "B748",
    "A388": "A380",
    "A333": "A330",
    "A332": "A330",
    "A339": "A330neo",
    "B733": "B737Classic",
    "B734": "B737Classic",
    "B763": "B767",
    "B38M": "B737MAX",
    "E290": "E2",
    "E295": "E2",
    "GLF4": "G450",
    "GLF6": "G650",
    "GLEX": "GLEX",
    "FA8X": "FA8X",
    "CL60": "CL60",
    None: "Unknown",
}


class DataProcessor(ABC):
    @abstractmethod
    def preprocess_data(self, obs: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def postprocess_data(self, **kwargs):
        raise NotImplementedError()


class TransformerProcessor(DataProcessor):
    def preprocess_data(self, obs: np.ndarray) -> Tuple[Tensor, Tensor]:
        obs = Tensor(obs).reshape((1, AIRCRAFT_COUNT, -1))

        return obs[:,:,:-1], obs[:,:,-1]

    def postprocess_data(self, action: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        action = action.squeeze(0)[attention_mask.squeeze(0).to(torch.int32) == 1]
        action = torch.hstack((action[:,:72].argmax(dim=1).unsqueeze(-1), action[:,72:])).numpy()
        action[:,0] = action[:,0]
        action[:,1] = np.round(action[:,1] * 16)
        action[:,2] = np.round(action[:,2] * 10 + 22)
        action = np.hstack((action, action[:,3:6].any(axis=1, keepdims=True)))
        action = np.vstack((action, np.zeros((AIRCRAFT_COUNT - action.shape[0], action.shape[1]))))

        return action.reshape(1, -1).astype(np.int32)


class GNNProcessor(DataProcessor):
    def preprocess_data(self, obs: np.ndarray) -> Data:
        obs = Tensor(obs).reshape((AIRCRAFT_COUNT, -1))
        obs = obs[obs[:,-1] == 1,:-1]
        # print(obs.shape)

        node_count = obs.shape[0]

        edge_index = np.array([(i, j) for j in range(node_count) for i in range(node_count)]).transpose(0, 1)
        edge_pos_0 = np.vstack((obs[edge_index[:,0], 2], obs[edge_index[:,0], 3])).transpose()
        edge_pos_1 = np.vstack((obs[edge_index[:,1], 2], obs[edge_index[:,1], 3])).transpose()
        edge_v_0 = np.vstack((obs[edge_index[:,0], 6], obs[edge_index[:,0], 7])).transpose()
        edge_v_1 = np.vstack((obs[edge_index[:,1], 6], obs[edge_index[:,1], 7])).transpose()
        delta_pos = edge_pos_0 - edge_pos_1
        v_sum = edge_v_1 - edge_v_0

        # Put distance, closure rate in edge_attr
        # Closure rate is defined as (pos2 - pos1) dot (v1 - v2) / norm(pos2 - pos1)
        pos_dist = np.linalg.norm(delta_pos, axis=1)
        edge_attr = np.vstack((
            pos_dist / np.sqrt(8),
            # Divide function call to handle when elements of pos_dist == 0
            np.divide(np.vecdot(delta_pos, v_sum), pos_dist, out=np.zeros_like(pos_dist), where=pos_dist != 0) / 2
        )).transpose()
        # edge_attr_old = torch.tensor([(
        #     np.sqrt(np.square(x[i, "x"] - x[j, "x"]) + np.square(x[i, "y"] - x[j, "y"]))
        # ) for i, j in edge_index_old], dtype=torch.float32) / np.sqrt(8)
        edge_index = torch.tensor(edge_index.transpose())
        edge_attr = torch.tensor(edge_attr).to(torch.float32)

        return Data(x=Tensor(obs).to(torch.float32), edge_index=edge_index, edge_attr=edge_attr)

    def postprocess_data(self, action: torch.Tensor) -> np.ndarray:
        # print(action)
        action = torch.hstack((action[:,:72].argmax(dim=1).unsqueeze(-1), action[:,72:])).numpy()
        action[:,0] = action[:,0]
        action[:,1] = np.round(action[:,1] * 16)
        action[:,2] = np.round(action[:,2] * 10 + 22)
        action = np.hstack((action[:,:3], action[:,3:6].any(axis=1, keepdims=True)))
        action = np.vstack((action, np.zeros((AIRCRAFT_COUNT - action.shape[0], action.shape[1]))))

        return action.reshape(1, -1).astype(np.int32)
