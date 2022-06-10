import os
import json


class CaseResizeInfo:

    def __init__(self, path='./configuration_files', filename='cases_scale_info.json'):
        self._full_path = os.path.join(path, filename)

    def save_case_scale_info(self, case_name, shape_before_delimit, shape_before_resize, original_shape, positions):

        is_resized = False

        if shape_before_resize != shape_before_delimit:
            is_resized = True

        info = {
            "shape_before_delimit": shape_before_delimit,
            "shape_before_resize": shape_before_resize,
            "original_shape": original_shape,
            "is_resized": is_resized,
            "positions": positions
        }

        infos = {}

        if os.path.isfile(self._full_path):
            with open(self._full_path) as json_file:
                infos = json.load(json_file)

        infos[case_name] = info

        with open(self._full_path, 'w') as fp:
            json.dump(infos, fp)

        return tuple(info["shape_before_delimit"]), tuple(info["shape_before_resize"]), tuple(info["original_shape"]), info["is_resized"], info["positions"]    

    def load_case_scale_info(self, case_name):
        infos = {}
        if os.path.isfile(self._full_path):
            with open(self._full_path) as json_file:
                infos = json.load(json_file)

        if not case_name in infos:
            return None, None, None, None, None

        info = infos[case_name]

        return tuple(info["shape_before_delimit"]), tuple(info["shape_before_resize"]), tuple(info["original_shape"]), info["is_resized"], info["positions"]


if __name__ == "__main__":

    cri = CaseResizeInfo()

    case_name = "case_00123"

    min_x, max_x, min_y, max_y, min_z, max_z = (0, 0, 0, 0, 0, 0)

    cri.save_case_scale_info(case_name, (0, 0), (0, 0), False,  [
        min_x, max_x, min_y, max_y, min_z, max_z])

    print(cri.load_case_scale_info(
        case_name))
