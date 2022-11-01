#!/usr/bin/env python3

import os
import shutil
import struct
import sys
from collections import abc
from typing import List, Tuple

from io_scene_vrm.common.gltf import pack_glb, parse_glb


def main() -> int:
    if len(sys.argv) < 2:
        print("Missing argv[1]")
        return 1
    enable_vertex_color = False
    with open(sys.argv[1], "rb") as f:
        read_bytes = f.read()
    json_dict, binary_chunk = parse_glb(read_bytes)

    if not isinstance(json_dict.get("buffers"), list):
        json_dict["buffers"] = []
    # position_buffer_index = len(json_dict["buffers"])
    position_buffer_byte_offset = len(binary_chunk)
    positions: List[Tuple[float, float, float]] = [
        (-0.1, 0.0, 0.1),
        (0.1, 0.0, 0.1),
        (0.0, 0.1, 0.0),
        #
        (0.0, 0.1, 0.0),
        (-0.1, 0.0, -0.1),
        (-0.1, 0.0, 0.1),
        #
        (0.0, 0.1, 0.0),
        (0.1, 0.0, -0.1),
        (-0.1, 0.0, -0.1),
        #
        (0.1, 0.0, 0.1),
        (0.1, 0.0, -0.1),
        (0.0, 0.1, 0.0),
        #
        (0.0, -0.1, 0.0),
        (0.1, 0.0, 0.1),
        (-0.1, 0.0, 0.1),
        #
        (-0.1, 0.0, 0.1),
        (-0.1, 0.0, -0.1),
        (0.0, -0.1, 0.0),
        #
        (-0.1, 0.0, -0.1),
        (0.1, 0.0, -0.1),
        (0.0, -0.1, 0.0),
        #
        (0.0, -0.1, 0.0),
        (0.1, 0.0, -0.1),
        (0.1, 0.0, 0.1),
    ]

    flat_positions = sum(positions, tuple())
    position_buffer_bytes = struct.pack(f"<{len(flat_positions)}f", *flat_positions)
    binary_chunk += position_buffer_bytes
    # json_dict["buffers"].append(
    #    {
    #        "uri": "data:application/gltf-buffer;base64,"
    #        + base64.b64encode(position_buffer_bytes).decode("ascii"),
    #        "byteLength": len(position_buffer_bytes),
    #    }
    # )

    texcoord_buffer_byte_offset = len(binary_chunk)
    texcoords: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        #
    ]
    assert len(texcoords) == len(positions)
    flat_texcoords = sum(texcoords, tuple())
    assert len(flat_texcoords) == len(flat_positions) / 3 * 2
    texcoord_buffer_bytes = struct.pack(f"<{len(flat_texcoords)}f", *flat_texcoords)
    binary_chunk += texcoord_buffer_bytes

    if enable_vertex_color:
        # color_buffer_index = len(json_dict["buffers"])
        color_buffer_byte_offset = len(binary_chunk)
        colors = list(
            map(
                lambda position: tuple(map(lambda c: 0.0 if c <= 0 else 1.0, position)),
                positions,
            )
        )

        assert len(colors) == len(positions)
        flat_colors = sum(colors, tuple())
        assert len(flat_colors) == len(flat_positions)
        color_buffer_bytes = struct.pack(f"<{len(flat_colors)}f", *flat_colors)
        binary_chunk += color_buffer_bytes
        # json_dict["buffers"].append(
        #    {
        #        "uri": "data:application/gltf-buffer;base64,"
        #        + base64.b64encode(color_buffer_bytes).decode("ascii"),
        #        "byteLength": len(color_buffer_bytes),
        #    }
        # )

    if not isinstance(json_dict.get("bufferViews"), list):
        json_dict["bufferViews"] = []
    position_buffer_view_index = len(json_dict["bufferViews"])
    json_dict["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": position_buffer_byte_offset,
            "byteLength": len(position_buffer_bytes),
            "target": 34962,  # ARRAY_BUFFER
        }
    )
    texcoord_buffer_view_index = len(json_dict["bufferViews"])
    json_dict["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": texcoord_buffer_byte_offset,
            "byteLength": len(texcoord_buffer_bytes),
            "target": 34962,  # ARRAY_BUFFER
        }
    )

    if enable_vertex_color:
        color_buffer_view_index = len(json_dict["bufferViews"])
        json_dict["bufferViews"].append(
            {
                "buffer": 0,
                "byteOffset": color_buffer_byte_offset,
                "byteLength": len(color_buffer_bytes),
                "target": 34962,  # ARRAY_BUFFER
            }
        )

    if not isinstance(json_dict.get("accessors"), list):
        json_dict["accessors"] = []
    position_accessors_index = len(json_dict["accessors"])
    json_dict["accessors"].append(
        {
            "bufferView": position_buffer_view_index,
            "byteOffset": 0,
            "type": "VEC3",
            "componentType": 5126,  # GL_FLOAT
            "count": len(positions),
            "min": [-0.1, -0.1, -0.1],
            "max": [0.1, 0.1, 0.1],
        }
    )
    texcoord_accessors_index = len(json_dict["accessors"])
    json_dict["accessors"].append(
        {
            "bufferView": texcoord_buffer_view_index,
            "byteOffset": 0,
            "type": "VEC2",
            "componentType": 5126,  # GL_FLOAT
            "count": len(texcoords),
            "min": [0.0, 0.0],
            "max": [1.0, 1.0],
        }
    )

    with open(os.path.join(os.path.dirname(__file__), "vrm1_test_texture.png"), "rb") as f:
        image_bytes = f.read()
    padded_image_bytes = bytes(image_bytes)
    while len(padded_image_bytes) % 16 == 0:
        padded_image_bytes += b'\x00'
    image_buffer_byte_offset = len(binary_chunk)
    binary_chunk += padded_image_bytes
    image_buffer_view_index = len(json_dict["bufferViews"])
    json_dict["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": image_buffer_byte_offset,
            "byteLength": len(image_bytes),
        }
    )

    if not isinstance(json_dict.get("samplers"), list):
        json_dict["samplers"] = []
    image_sampler_index = len(json_dict["samplers"])
    json_dict["samplers"].append({
            "magFilter": 9729,
            "minFilter": 9987,
            "wrapS": 10497,
            "wrapT": 10497
    })

    if not isinstance(json_dict.get("images"), list):
        json_dict["images"] = []
    image_index = len(json_dict["images"])
    json_dict["images"].append({
        "bufferView": image_buffer_view_index,
        "mimeType": "image/png",
    })

    if not isinstance(json_dict.get("textures"), list):
        json_dict["textures"] = []
    texture_index = len(json_dict["textures"])
    json_dict["textures"].append({
        "sampler": image_sampler_index,
        "source": image_index,
    })

    if not isinstance(json_dict.get("materials"), list):
        json_dict["materials"] = []
    material_index = len(json_dict["materials"])
    json_dict["materials"].append(
        {
            "pbrMetallicRoughness": {"baseColorTexture": {"index": texture_index}},
            "emissiveFactor": [0.2, 0.2, 0.2]
        }
    )

    if enable_vertex_color:
        color_accessors_index = len(json_dict["accessors"])
        json_dict["accessors"].append(
            {
                "bufferView": color_buffer_view_index,
                "byteOffset": 0,
                "type": "VEC3",
                "componentType": 5126,  # GL_FLOAT
                "count": len(colors),
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            }
        )
        primitive = {
            "attributes": {
                "POSITION": position_accessors_index,
                "COLOR_0": color_accessors_index,
                "TEXCOORD_0": texcoord_accessors_index,
            },
            "material": material_index
        }
    else:
        primitive = {
            "attributes": {
                "POSITION": position_accessors_index,
                "TEXCOORD_0": texcoord_accessors_index,
            },
            "material": material_index
        }

    meshes = json_dict.get("meshes")
    if not isinstance(meshes, list):
        meshes = []
        json_dict["meshes"] = meshes
    mesh_index = len(meshes)
    meshes.append({"primitives": [primitive]})

    nodes = json_dict.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        json_dict["nodes"] = nodes

    human_bones_dict = {}

    hips_index = len(nodes)
    spine_index = len(nodes) + 1
    head_index = len(nodes) + 2

    left_upper_leg_index = len(nodes) + 3
    left_lower_leg_index = len(nodes) + 4
    left_foot_index = len(nodes) + 5
    left_upper_arm_index = len(nodes) + 6
    left_lower_arm_index = len(nodes) + 7
    left_hand_index = len(nodes) + 8

    right_upper_leg_index = len(nodes) + 9
    right_lower_leg_index = len(nodes) + 10
    right_foot_index = len(nodes) + 11
    right_upper_arm_index = len(nodes) + 12
    right_lower_arm_index = len(nodes) + 13
    right_hand_index = len(nodes) + 14

    nodes.extend(
        [
            {
                "name": "hips",
                "children": [
                    spine_index,
                    left_upper_leg_index,
                    right_upper_leg_index,
                ],
                "translation": [0, 0.5, 0],
            },
            {
                "name": "spine",
                "children": [head_index, left_upper_arm_index, right_upper_arm_index],
                "translation": [0, 0.25, 0],
                "mesh": mesh_index,
            },
            {"name": "head", "translation": [0, 0.5, 0]},
            {
                "name": "leftUpperLeg",
                "children": [left_lower_leg_index],
                "translation": [0.125, 0, 0],
                "mesh": mesh_index,
            },
            {
                "name": "leftLowerLeg",
                "children": [left_foot_index],
                "translation": [0, -0.25, 0],
                "mesh": mesh_index,
            },
            {"name": "leftFoot", "translation": [0, -0.25, 0], "mesh": mesh_index},
            {
                "name": "leftUpperArm",
                "children": [left_lower_arm_index],
                "translation": [0.25, 0.25, 0],
                "mesh": mesh_index,
            },
            {
                "name": "leftLowerArm",
                "children": [left_hand_index],
                "translation": [0.25, 0, 0],
                "mesh": mesh_index,
            },
            {"name": "leftHand", "translation": [0.25, 0, 0], "mesh": mesh_index},
            {
                "name": "rightUpperLeg",
                "children": [right_lower_leg_index],
                "translation": [-0.125, 0, 0],
                "mesh": mesh_index,
            },
            {
                "name": "rightLowerLeg",
                "children": [right_foot_index],
                "translation": [0, -0.25, 0],
                "mesh": mesh_index,
            },
            {"name": "rightFoot", "translation": [0, -0.25, 0], "mesh": mesh_index},
            {
                "name": "rightUpperArm",
                "children": [right_lower_arm_index],
                "translation": [-0.25, 0.25, 0],
                "mesh": mesh_index,
            },
            {
                "name": "rightLowerArm",
                "children": [right_hand_index],
                "translation": [-0.25, 0, 0],
                "mesh": mesh_index,
            },
            {"name": "rightHand", "translation": [-0.25, 0, 0], "mesh": mesh_index},
        ]
    )

    scenes = json_dict.get("scenes")
    scene = json_dict.get("scene")
    if (
        isinstance(scene, int)
        and isinstance(scenes, abc.Collection)
        and len(scenes) > scene
        and isinstance(scenes[scene], dict)
        and isinstance(scenes[scene].get("nodes"), abc.Collection)
    ):
        if not isinstance(nodes[head_index].get("children"), abc.Collection):
            nodes[head_index]["children"] = []
        nodes[head_index]["children"].extend(scenes[scene]["nodes"])
        json_dict["scenes"][scene] = {"nodes": [hips_index]}
    else:
        json_dict["scenes"] = [{"nodes": [hips_index]}]

    human_bones_dict = {
        "hips": {"node": hips_index},
        "spine": {"node": spine_index},
        "head": {"node": head_index},
        "leftUpperLeg": {"node": left_upper_leg_index},
        "leftLowerLeg": {"node": left_lower_leg_index},
        "leftFoot": {"node": left_foot_index},
        "rightUpperLeg": {"node": right_upper_leg_index},
        "rightLowerLeg": {"node": right_lower_leg_index},
        "rightFoot": {"node": right_foot_index},
        "leftUpperArm": {"node": left_upper_arm_index},
        "leftLowerArm": {"node": left_lower_arm_index},
        "leftHand": {"node": left_hand_index},
        "rightUpperArm": {"node": right_upper_arm_index},
        "rightLowerArm": {"node": right_lower_arm_index},
        "rightHand": {"node": right_hand_index},
    }

    extensions_used = json_dict.get("extensionsUsed")
    if not isinstance(extensions_used, list):
        extensions_used = []
        json_dict["extensionsUsed"] = extensions_used
    if "VRMC_vrm" not in extensions_used:
        extensions_used.append("VRMC_vrm")

    extensions = json_dict.get("extensions")
    if not isinstance(extensions, dict):
        extensions = {}
        json_dict["extensions"] = extensions
    extensions["VRMC_vrm"] = {
        "specVersion": "1.0",
        "meta": {
            "licenseUrl": "https://vrm.dev/licenses/1.0/",
            "name": os.path.basename(sys.argv[1]),
            "version": "UniVRMへのバグ報告用",
            "avatarPermission": "onlyAuthor",
            "allowExcessivelyViolentUsage": False,
            "allowExcessivelySexualUsage": False,
            "commercialUsage": "personalNonProfit",
            "allowPoliticalOrReligiousUsage": False,
            "allowAntisocialOrHateUsage": False,
            "creditNotation": "required",
            "allowRedistribution": False,
            "modification": "prohibited",
            "authors": [
                "https://github.com/saturday06",
                "https://github.com/KhronosGroup/glTF-Sample-Models/tree/b4c124c18171b6dead0350b6e46826e320a49a23/2.0",
            ],
            "otherLicenseUrl": "https://github.com/KhronosGroup/glTF-Sample-Models/tree/b4c124c18171b6dead0350b6e46826e320a49a23/2.0/",
            "thirdPartyLicenses": "こちらはUniVRMへのバグ報告を目的としたモデルのため一般利用はお控えください"
        },
        "humanoid": {"humanBones": human_bones_dict},
        "firstPerson": {},
        "lookAt": {
            "offsetFromHeadBone": [0, 0, 0],
            "type": "bone",
            "rangeMapHorizontalInner": {"inputMaxValue": 0, "outputScale": 0},
            "rangeMapHorizontalOuter": {"inputMaxValue": 0, "outputScale": 0},
            "rangeMapVerticalDown": {"inputMaxValue": 0, "outputScale": 0},
            "rangeMapVerticalUp": {"inputMaxValue": 0, "outputScale": 0},
        },
        "expressions": {
            "preset": {
                "happy": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "angry": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "sad": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "relaxed": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "surprised": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "neutral": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "aa": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "ih": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "ou": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "ee": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "oh": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "blink": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "blinkLeft": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "blinkRight": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "lookUp": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "lookDown": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "lookLeft": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
                "lookRight": {
                    "isBinary": False,
                    "overrideBlink": "none",
                    "overrideLookAt": "none",
                    "overrideMouth": "none",
                },
            },
            "custom": {},
        },
    }

    json_dict["buffers"][0]["byteLength"] = len(binary_chunk)
    vrm1_bytes = pack_glb(json_dict, binary_chunk)
    with open(sys.argv[1] + ".vrm", "wb") as f:
        f.write(vrm1_bytes)
    shutil.copy(sys.argv[1] + ".vrm", sys.argv[1] + ".vrm.glb")

    return 0


if __name__ == "__main__":
    sys.exit(main())
