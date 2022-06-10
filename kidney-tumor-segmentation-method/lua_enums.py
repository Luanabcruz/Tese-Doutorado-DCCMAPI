from enum import Enum


class DataAug(Enum):
    OFFLINE = 'Offline'
    ONLINE = 'Online (Real Time)'
    NONE = 'Sem Augmentation'
