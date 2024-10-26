from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple

class BaseGTALabels(metaclass=ABCMeta):
    pass

@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))
    dynamic = GTA5Label(ID=19, color=(111, 74, 0))
    ground = GTA5Label(ID=20, color=(81, 0, 81))
    parking = GTA5Label(ID=21, color=(250, 170, 160))
    rail_track = GTA5Label(ID=22, color=(230, 150, 140))
    bridge = GTA5Label(ID=23, color=(150, 100, 100))
    tunnel = GTA5Label(ID=24, color=(150, 120, 90))
    polegroup = GTA5Label(ID=25, color=(100, 58, 200))
    caravan = GTA5Label(ID=26, color=(0, 0, 90))
    trailer = GTA5Label(ID=27, color=(0, 0, 110))
    unlabeled = GTA5Label(ID=28, color=(0, 0, 0))
    ego_vehicle = GTA5Label(ID=29, color=(0, 0, 0))
    rectification_border = GTA5Label(ID=30, color=(0, 0, 0))
    out_of_roi = GTA5Label(ID=31, color=(0, 0, 0))
    static = GTA5Label(ID=32, color=(0, 0, 0))
    licence_plate = GTA5Label(ID = 33, color = (40, 40, 40))
   
    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
        unlabeled,
        ego_vehicle, 
        rectification_border, 
        out_of_roi, 
        static, 
        dynamic, 
        ground,
        parking, 
        rail_track, 
        bridge, 
        tunnel, 
        polegroup,
        caravan, 
        trailer, 
        licence_plate, 
      ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret
