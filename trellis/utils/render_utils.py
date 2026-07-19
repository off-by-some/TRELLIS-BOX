import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence
from .device import is_cuda_runtime


def _rotation_matrix(yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rx @ rz


def _prepare_mesh_arrays(mesh: MeshExtractResult, max_faces=12000):
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().long().cpu().numpy()
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    scale = scale if scale > 1e-6 else 1.0
    vertices = (vertices - center) / scale * 2.0
    return vertices.astype(np.float32), faces.astype(np.int64)


def _software_mesh_frame(vertices, faces, yaw, pitch, resolution, mode):
    image = np.zeros((resolution, resolution, 3), dtype=np.float32)
    zbuf = np.full((resolution, resolution), np.inf, dtype=np.float32)

    rot = _rotation_matrix(yaw, pitch)
    verts = vertices @ rot.T
    xy = verts[:, :2] * 0.9
    pixels = np.empty_like(xy)
    pixels[:, 0] = (xy[:, 0] + 1.0) * 0.5 * (resolution - 1)
    pixels[:, 1] = (1.0 - (xy[:, 1] + 1.0) * 0.5) * (resolution - 1)
    depth = -verts[:, 2]

    tri = pixels[faces]
    tri_depth = depth[faces]
    face_vertices = verts[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-6)
    light = np.array([0.25, -0.45, 0.86], dtype=np.float32)
    light = light / np.linalg.norm(light)

    for i in range(faces.shape[0]):
        pts = tri[i]
        min_xy = np.floor(pts.min(axis=0)).astype(np.int32)
        max_xy = np.ceil(pts.max(axis=0)).astype(np.int32)
        min_x, min_y = np.maximum(min_xy, 0)
        max_x, max_y = np.minimum(max_xy, resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        p0, p1, p2 = pts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-6:
            continue

        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue

        z = w0 * tri_depth[i, 0] + w1 * tri_depth[i, 1] + w2 * tri_depth[i, 2]
        current = zbuf[min_y:max_y + 1, min_x:max_x + 1]
        update = mask & (z < current)
        if not update.any():
            continue

        normal = normals[i]
        if mode == "normal":
            color = normal * 0.5 + 0.5
        else:
            shade = max(float(normal @ light), 0.0) * 0.75 + 0.25
            color = np.array([0.78, 0.82, 0.88], dtype=np.float32) * shade

        patch = image[min_y:max_y + 1, min_x:max_x + 1]
        patch[update] = color
        current[update] = z[update]

    return np.clip(image * 255, 0, 255).astype(np.uint8)


def _render_mesh_video_software(sample, resolution=384, num_frames=36):
    vertices, faces = _prepare_mesh_arrays(sample)
    yaws = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    pitchs = 0.25 + 0.35 * np.sin(np.linspace(0, 2 * np.pi, num_frames, endpoint=False))
    color = [_software_mesh_frame(vertices, faces, y, p, resolution, "color") for y, p in zip(yaws, pitchs)]
    normal = [_software_mesh_frame(vertices, faces, y, p, resolution, "normal") for y, p in zip(yaws, pitchs)]
    return {"color": color, "normal": normal}


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    if isinstance(sample, Octree):
        from ..renderers import OctreeRenderer
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        from ..renderers import GaussianRenderer
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        from ..renderers import MeshRenderer
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            # Handle NaN/Inf values in normal maps from fp16 precision
            normal_array = res['normal'].detach().cpu().numpy().transpose(1, 2, 0)
            normal_array = np.nan_to_num(normal_array, nan=0.0, posinf=1.0, neginf=0.0)
            rets['normal'].append(np.clip(normal_array * 255, 0, 255).astype(np.uint8))
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    if isinstance(sample, MeshExtractResult) and not is_cuda_runtime():
        return _render_mesh_video_software(sample, resolution=min(resolution, 384), num_frames=min(num_frames, 36))
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
