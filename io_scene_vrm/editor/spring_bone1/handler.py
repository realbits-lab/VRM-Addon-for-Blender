import math
from typing import Callable

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Quaternion, Vector

from ...common.logging import get_logger
from .property_group import SpringBone1JointPropertyGroup

logger = get_logger(__name__)

if not persistent:  # for fake-bpy-modules

    def persistent(_func: Callable[[object], None]) -> Callable[[object], None]:
        raise NotImplementedError


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
        for (head_spring_joint, tail_spring_joint) in zip(
            spring.joints, spring.joints[1:]
        ):
            update_spring_joint_pair(
                delta_time, obj, head_spring_joint, tail_spring_joint
            )


def update_spring_joint_pair(
    delta_time: float,
    obj: bpy.types.Object,
    head_spring_joint: SpringBone1JointPropertyGroup,
    tail_spring_joint: SpringBone1JointPropertyGroup,
) -> None:
    head_bone_name = head_spring_joint.node.value
    head_pose_bone = obj.pose.bones.get(head_bone_name)
    if not head_pose_bone:
        return
    if head_pose_bone.bone.use_connect:
        logger.warning(
            f'Head "{head_bone_name}" is connected with parent. Not implemented yet.'
        )
        return
    if not head_pose_bone.bone.use_local_location:
        logger.warning(
            f'Head "{head_bone_name}" is not local location. Not implemented yet.'
        )
        return

    tail_bone_name = tail_spring_joint.node.value
    tail_pose_bone = obj.pose.bones.get(tail_bone_name)
    if not tail_pose_bone:
        return
    if tail_pose_bone.bone.use_connect:
        logger.warning(
            f'Tail "{tail_bone_name}" is connected with parent. Not implemented yet.'
        )
        return
    if not tail_pose_bone.bone.use_local_location:
        logger.warning(
            f'Tail "{tail_bone_name}" is not local location. Not implemented yet.'
        )
        return

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

    head_rest_bone_matrix_world = Matrix(
        obj.matrix_world
    ) @ head_pose_bone.bone.convert_local_to_pose(
        matrix=Matrix(),
        matrix_local=head_pose_bone.bone.matrix_local,
        parent_matrix=head_pose_bone.parent.matrix,
        parent_matrix_local=head_pose_bone.parent.bone.matrix_local,
    )

    tail_rest_bone_matrix_world = Matrix(
        obj.matrix_world
    ) @ tail_pose_bone.bone.convert_local_to_pose(
        matrix=Matrix(),
        matrix_local=tail_pose_bone.bone.matrix_local,
        parent_matrix=head_pose_bone.parent.matrix,
        parent_matrix_local=head_pose_bone.parent.bone.matrix_local,
    )

    # https://github.com/vrm-c/vrm-specification/blob/993a90a5bda9025f3d9e2923ad6dea7506f88553/specification/VRMC_springBone-1.0/README.ja.md#%E5%88%9D%E6%9C%9F%E5%8C%96
    current_tail = (
        Matrix(obj.matrix_world) @ Matrix(tail_pose_bone.matrix)
    ).to_translation()

    # boneAxisとboneLengthの仕様の記載が間違っていそう。
    # > prevTail ・ currentTail は、そのJointが対象とする子Nodeの、ワールド空間における位置を表します。
    # という記載はTailは「子Node」であると示されるが
    # > boneAxis は、そのJointが対象とする子Nodeの、ローカル空間におけるレスト状態の伸びる方向を表します。
    # という記載における「子Node」は、Headノードでないと疑似コード
    # > // 長さの制約
    # > nextTail = worldPosition + (nextTail - worldPosition).normalized * boneLength;
    # におけるコメント、ロジックの趣旨と合致しない。

    head_rest_to_tail_rest_translation_world = (
        tail_rest_bone_matrix_world.to_translation()
        - head_rest_bone_matrix_world.to_translation()
    )
    bone_axis = head_rest_to_tail_rest_translation_world.normalized()
    bone_length = head_rest_to_tail_rest_translation_world.length

    # if not head_spring_joint.state.initialized:
    if bpy.context.scene.frame_current == 1:
        logger.warning("Initialized")
        head_spring_joint.state.prev_tail = current_tail[:]
        head_spring_joint.state.initialized = True
    prev_tail = Vector(head_spring_joint.state.prev_tail)

    # https://github.com/vrm-c/vrm-specification/blob/993a90a5bda9025f3d9e2923ad6dea7506f88553/specification/VRMC_springBone-1.0/README.ja.md#%E6%85%A3%E6%80%A7%E8%A8%88%E7%AE%97
    world_position = (
        Matrix(obj.matrix_world) @ Matrix(head_pose_bone.matrix)
    ).to_translation()
    local_rotation = Quaternion(head_pose_bone.rotation_quaternion)
    parent_world_rotation = (
        Matrix(obj.matrix_world) @ Matrix(head_pose_bone.parent.matrix)
        if head_pose_bone.parent
        else Matrix()
    ).to_quaternion()

    # calculate the next tail position using verlet integration
    inertia = (current_tail - prev_tail) * (1.0 - head_spring_joint.drag_force)
    stiffness = (
        delta_time
        * (parent_world_rotation @ local_rotation @ bone_axis)
        * head_spring_joint.stiffness
    )
    external = (
        delta_time
        * Vector(head_spring_joint.gravity_dir)
        * head_spring_joint.gravity_power
    )

    next_tail = current_tail + inertia + stiffness + external

    # constrain the length
    next_tail = world_position + (next_tail - world_position).normalized() * bone_length

    # update prevTail and currentTail
    prev_tail = current_tail
    current_tail = next_tail

    # update rotation
    t = (next_tail - head_rest_bone_matrix_world.to_translation()).normalized()
    f = (
        tail_rest_bone_matrix_world.to_translation()
        - head_rest_bone_matrix_world.to_translation()
    ).normalized()
    c = f.cross(t)
    a = math.acos(max(min(f.dot(t), 1), -1))
    c.rotate(head_rest_bone_matrix_world.to_quaternion().inverted())
    q = Quaternion(c, a)

    if head_pose_bone.rotation_mode != "QUATERNION":
        head_pose_bone.rotation_mode = "QUATERNION"
    head_pose_bone.rotation_quaternion = q
    head_spring_joint.state.prev_tail = prev_tail[:]


@persistent  # type: ignore[misc]
def frame_change_pre(_dummy: object) -> None:
    delta_time = float(bpy.context.scene.render.fps_base) / float(
        bpy.context.scene.render.fps
    )
    update_objects(delta_time)
