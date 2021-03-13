from typing import Dict, Tuple, Optional, List
import uuid
import struct
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class IntListPropertiesChannel(SideChannel):
    """
    This is the SideChannel for Int list properties shared with Unity.
    You can modify the Int list properties of an environment with the commands
    set_property , get_property and list_properties.
    """

    def __init__(self, channel_id: uuid.UUID = None) -> None:
        self._int_list_properties: Dict[str, List[int]] = {}
        if channel_id is None:
            channel_id = uuid.UUID(("d554ba67-03ee-459e-974a-1cba23324c04"))
        super().__init__(channel_id)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        k, v = self.deserialize_int_list_prop(msg.get_raw_bytes())
        self._int_list_properties[k] = v

    def set_property(self, key: str, value: List[int]) -> None:
        """
        Sets a property in the Unity Environment.
        :param key: The string identifier of the property.
        :param value: The int32 list of the property.
        """
        self._int_list_properties[key] = value
        super().queue_message_to_send(self.serialize_int_list_prop(key, value))

    def get_property(self, key: str) -> Optional[List[int]]:
        """
        Gets a property in the Unity Environment. If the property was not
        found, will return None.
        :param key: The string identifier of the property.
        :return: The float value of the property or None.
        """
        return self._int_list_properties.get(key)

    def list_properties(self) -> List[str]:
        """
        Returns a list of all the string identifiers of the properties
        currently present in the Unity Environment.
        """
        return list(self._int_list_properties.keys())

    def get_property_dict_copy(self) -> Dict[str, List[int]]:
        """
        Returns a copy of the int list properties.
        :return:
        """
        return dict(self._int_list_properties)

    @staticmethod
    def serialize_int_list_prop(key: str, value: List[int]) -> OutgoingMessage:
        result = bytearray()
        encoded_key = key.encode("ascii")
        result += struct.pack("<i", len(encoded_key))
        result += encoded_key
        result += struct.pack("<i", len(value))

        for v in value:
            result += struct.pack("<I", v)
        msg = OutgoingMessage()
        msg.set_raw_bytes(result)
        return msg

    @staticmethod
    def deserialize_int_list_prop(data: bytes) -> Tuple[str, List[int]]:
        offset = 0
        encoded_key_len = struct.unpack_from("<i", data, offset)[0]
        offset = offset + 4
        key = data[offset : offset + encoded_key_len].decode("ascii")
        offset = offset + encoded_key_len

        encoded_value_len = struct.unpack_from("<i", data, offset)[0]
        offset = offset + 4
        value = []
        for i in list(range(encoded_value_len)):
            value.append(struct.unpack_from("<I", data, offset+i*4)[0])
        return key, value
    