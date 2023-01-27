import math
from typing import Callable, List, Tuple

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Quaternion, Vector

from ...common.logging import get_logger
from .property_group import SpringBone1JointPropertyGroup

logger = get_logger(__name__)

if not persistent:  # for fake-bpy-modules

    def persistent(_func: Callable[[object], None]) -> Callable[[object], None]:
        raise NotImplementedError


global_counts = [0]
global_initialized = [False]


# https://github.com/vrm-c/vrm-specification/tree/993a90a5bda9025f3d9e2923ad6dea7506f88553/specification/VRMC_springBone-1.0#update-procedure
def update_objects(delta_time: float) -> None:
    for obj in bpy.data.objects:
        update_object(delta_time, obj)


def update_object(delta_time: float, obj: bpy.types.Object) -> None:
    if obj.type != "ARMATURE":
        return
    ext = obj.data.vrm_addon_extension
    if not ext.is_vrm1():
        return
    spring_bone1 = ext.spring_bone1
    for spring in spring_bone1.springs:
        inputs: List[
            Tuple[
                SpringBone1JointPropertyGroup,
                bpy.types.PoseBone,
                SpringBone1JointPropertyGroup,
                bpy.types.PoseBone,
                float,
                Vector,
            ]
        ] = []
        for (head_spring_joint, tail_spring_joint) in zip(
            spring.joints, spring.joints[1:]
        ):
            head_bone_name = head_spring_joint.node.value
            head_pose_bone = obj.pose.bones.get(head_bone_name)
            if not head_pose_bone:
                continue
            if head_pose_bone.bone.use_connect:
                logger.warning(
                    f'Head "{head_bone_name}" is connected with parent. Not implemented yet.'
                )
                continue
            if not head_pose_bone.bone.use_local_location:
                logger.warning(
                    f'Head "{head_bone_name}" is not local location. Not implemented yet.'
                )
                continue

            tail_bone_name = tail_spring_joint.node.value
            tail_pose_bone = obj.pose.bones.get(tail_bone_name)
            if not tail_pose_bone:
                continue
            if tail_pose_bone.bone.use_connect:
                logger.warning(
                    f'Tail "{tail_bone_name}" is connected with parent. Not implemented yet.'
                )
                continue
            if not tail_pose_bone.bone.use_local_location:
                logger.warning(
                    f'Tail "{tail_bone_name}" is not local location. Not implemented yet.'
                )
                continue

            head_tail_parented = False
            searching_tail_parent = tail_pose_bone.parent
            while searching_tail_parent:
                if searching_tail_parent.name == head_bone_name:
                    head_tail_parented = True
                    break
                searching_tail_parent = searching_tail_parent.parent
            if not head_tail_parented:
                logger.error(f'"{head_bone_name}" and "{tail_bone_name}" are not parented')
                return

            head_tail_world_distance = (
                (
                    Matrix(obj.matrix_world) @ Matrix(head_pose_bone.bone.matrix_local)
                ).to_translation()
                - (
                    Matrix(obj.matrix_world) @ Matrix(tail_pose_bone.bone.matrix_local)
                ).to_translation()
            ).length

            # https://github.com/vrm-c/vrm-specification/blob/993a90a5bda9025f3d9e2923ad6dea7506f88553/specification/VRMC_springBone-1.0/README.ja.md#%E5%88%9D%E6%9C%9F%E5%8C%96
            current_tail_world_location = (
                Matrix(obj.matrix_world) @ Matrix(tail_pose_bone.matrix)
            ).to_translation()

            inputs.append(
                (
                    head_spring_joint,
                    head_pose_bone,
                    tail_spring_joint,
                    tail_pose_bone,
                    head_tail_world_distance,
                    current_tail_world_location,
                )
            )

        for (
            head_spring_joint,
            head_pose_bone,
            tail_spring_joint,
            tail_pose_bone,
            head_tail_world_distance,
            current_tail_world_location,
        ) in inputs:
            update_spring_joint_pair(
                delta_time,
                obj,
                head_spring_joint,
                head_pose_bone,
                tail_spring_joint,
                tail_pose_bone,
                head_tail_world_distance,
                current_tail_world_location,
            )


def update_spring_joint_pair(
    delta_time: float,
    obj: bpy.types.Object,
    head_spring_joint: SpringBone1JointPropertyGroup,
    head_pose_bone: bpy.types.PoseBone,
    tail_spring_joint: SpringBone1JointPropertyGroup,
    tail_pose_bone: bpy.types.PoseBone,
    head_tail_world_distance: float,
    current_tail_world_location: Vector,
) -> None:
    # boneAxisとboneLengthの仕様の記載が間違っていそう。
    # > prevTail ・ currentTail は、そのJointが対象とする子Nodeの、ワールド空間における位置を表します。
    # という記載はTailは「子Node」であると示されるが
    # > boneAxis は、そのJointが対象とする子Nodeの、ローカル空間におけるレスト状態の伸びる方向を表します。
    # という記載における「子Node」は、Headノードでないと疑似コード
    # > // 長さの制約
    # > nextTail = worldPosition + (nextTail - worldPosition).normalized * boneLength;
    # におけるコメント、ロジックの趣旨と合致しない。

    # if not head_spring_joint.state.initialized:
    if not global_initialized[0]:
        logger.warning("Initialized")
        global_initialized[0] = True
        head_spring_joint.state.prev_tail = current_tail_world_location[:]
        head_spring_joint.state.initialized = True

    # prev_tail_world_location = Vector(head_spring_joint.state.prev_tail)

    external = (
        delta_time
        * Vector(head_spring_joint.gravity_dir)
        * head_spring_joint.gravity_power
    )

    next_head_world_location = Matrix(obj.matrix_world) @ Matrix(head_pose_bone.matrix).to_translation()
    next_tail_world_location = current_tail_world_location + external

    # constrain the length
    next_tail_world_location = next_head_world_location + (next_tail_world_location - next_head_world_location).normalized() * head_tail_world_distance

    # update prevTail and currentTail
    head_spring_joint.state.prev_tail = current_tail_world_location[:]

    # update rotation
    # t = (next_tail - head_rest_bone_matrix_world.to_translation()).normalized()
    # f = (
    #     tail_rest_bone_matrix_world.to_translation()
    #     - head_rest_bone_matrix_world.to_translation()
    # ).normalized()
    # c = f.cross(t)
    # a = math.acos(max(min(f.dot(t), 1), -1))
    # c.rotate(head_rest_bone_matrix_world.to_quaternion().inverted())
    # q = Quaternion(c, a)

    # if head_pose_bone.rotation_mode != "QUATERNION":
    #     head_pose_bone.rotation_mode = "QUATERNION"
    # head_pose_bone.rotation_quaternion = q
    # head_spring_joint.state.prev_tail = prev_tail[:]


@persistent  # type: ignore[misc]
def frame_change_pre(_dummy: object) -> None:
    # delta_time = float(bpy.context.scene.render.fps_base) / float(
    #     bpy.context.scene.render.fps
    # )
    # update_objects(delta_time)
    if global_counts[0] >= 1:
        return

    global_counts[0] += 1
    logger.warning(
        f"================= global_count={global_counts[0]} frame_current={bpy.context.scene.frame_current} ================="
    )

    update_objects(1)
