from typing import Any, Dict, List, Tuple


class DataClassManager:
    @staticmethod
    def remove_null_values(data):
        """Remove the None values from json output + remove json attributes"""
        return {k: v for (k, v) in data if v is not None}

    @staticmethod
    def replace_null_with_empty_string(data):
        """To replace the None values with empty strings + remove json attributes"""
        return {k: ("" if v is None else v) for (k, v) in data}

    @staticmethod
    def from_jsonarray_to_list(
        class_ref, json_list: List[Dict], list_tuples_json_class: List[Tuple[Any, Any]]
    ) -> List:  #
        """
        Transforme un json_list (
            [
                {k1 : V1, K2 : V2},
                {k3 : V3, K4 : V4}
            ]) en une list[ClassName] python
        Args:
            - `json_list`: List of Json à transformer
            - `ClassName`: qui est le nom de la class dont sera fait la list
            - `list_tuples_json_class`: une list de tuples pour matcher chaque key du json object
        avec l'attribue qui lui correspond dans la class ClassName : [(k1, attr1), (k2, attr2)]
        """
        list_instances: List[class_ref] = []
        attrs = [
            attr
            for attr in dir(class_ref)
            if not callable(getattr(class_ref, attr)) and not attr.startswith("__")
        ]
        tmp = {}
        for json_obj in json_list:
            for item in list_tuples_json_class:
                if item[1] not in attrs:
                    pass
                tmp[item[1]] = json_obj[item[0]]
            instance = class_ref(**tmp)
            list_instances.append(instance)

        return list_instances
