import os
import time
import copy
import open3d as o3d
import numpy as np
import flowiz as fz
import matplotlib.pyplot as plt
import torch


def visualize_outputs(images_dict, writer, iteration_number, title=''):
    show_optical_flow = True
    show_depth_pred = True
    show_meshes_pred = True

    ######## Visualize optical flow prediction #########
    if show_optical_flow:
        plot_optical_flow(images_dict,writer,title,iteration_number)

    ######## Visualize depth prediction #########
    if show_depth_pred:
        plot_predicted_depth_maps(images_dict,writer,title,iteration_number)

    ######## Visualize meshes prediction #########
    if show_meshes_pred:
        plot_point_clouds(images_dict,writer,title,iteration_number)
    writer.close()


def plot_optical_flow(images_dict,writer,title,iteration_number):
    rows = 1
    cols = 2
    gt_flow_img = fz.convert_from_flow(images_dict['optical_flow_gt'][0].permute(1, 2, 0).detach().cpu().numpy())
    pred_flow_img = fz.convert_from_flow(images_dict['optical_flow_pred'][0].permute(1, 2, 0).detach().cpu().numpy())

    flow_mask = images_dict['optical_flow_mask'][0][0, ...]  # flow_mask is duplicated across the feature dimension
    flow_mask_float = flow_mask.type(torch.int).detach().cpu().numpy()[..., None]

    f = plt.figure(figsize=(6, 4))
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)

    a = f.add_subplot(rows, cols, 1)
    a.set_title("gt_flow_img flow")
    a.set_axis_off()

    plt.imshow(gt_flow_img * flow_mask_float, cmap="jet")

    a = f.add_subplot(rows, cols, 2)
    a.set_axis_off()
    a.set_title("pred_flow_img flow")
    plt.imshow(pred_flow_img * flow_mask_float, cmap="jet")
    f.tight_layout()

    writer.add_figure(title + '_optical_flow', f, iteration_number, close=True)
def plot_point_clouds(images_dict,writer,title,iteration_number):

    ind = 0
    #
    source_point_cloud_vertices = images_dict['source_point_cloud']['vertices'].detach().cpu()
    source_point_cloud_colors = images_dict['source_point_cloud']['colors'].detach().cpu()
    #
    deformed_points_pred_vertices = images_dict['deformed_points_pred']['vertices'].detach().cpu()
    deformed_points_pred_colors = images_dict['deformed_points_pred']['colors']

    deformed_points_gt_vertices = images_dict['deformed_points_gt']['vertices'].detach().cpu()
    deformed_points_gt_colors = images_dict['deformed_points_gt']['colors']

    ### Plot using open3d
    # # pcls = [source_point_cloud_vertices[ind], deformed_points_pred_vertices[ind], deformed_points_gt_vertices[ind]]
    # # pcls_colors = [source_point_cloud_colors[ind], deformed_points_pred_colors[ind], deformed_points_gt_colors[ind]]
    # # plot_3d_data_debug(pcls=pcls, pcls_colors=pcls_colors)

    ### source_point_cloud
    p1 = source_point_cloud_vertices[ind].view(3, -1).permute(1, 0).unsqueeze(0)
    p1[...,1]*=-1
    p1[...,2]*=-1
    c1 =(source_point_cloud_colors[ind].view(3, -1).permute(1, 0).unsqueeze(0)*255).int()
    c1 = c1.type(torch.int64)

    ### deformed_points_pred_vertices
    p2 = deformed_points_pred_vertices[ind].unsqueeze(0)
    p2[...,1]*=-1
    p2[...,2]*=-1

    c2 = np.ones_like(p2, dtype=np.int32) * [0,255,255]
    c2 = torch.as_tensor(c2)

    ### deformed_points_gt
    p3 = deformed_points_gt_vertices[ind].unsqueeze(0)
    p3[...,1]*=-1
    p3[...,2]*=-1
    c3 = np.ones_like(p3, dtype=np.int32) * [255,0,0]
    c3 = torch.as_tensor(c3)
    pcls_tensor = torch.cat([p1,p2,p3], dim=1)
    colors_tensor = torch.cat([c1,c2,c3], dim=1)
    # pcls = torch.cat((p1,p2),dim=1)
    # colors_tensor  = torch.cat((c1,c2),dim=1)
    # writer.add_mesh('a', vertices=p1,colors=c1,global_step=iteration_number)

    writer.add_mesh(title, vertices=pcls_tensor,colors=colors_tensor ,global_step=iteration_number)


def plot_predicted_depth_maps(images_dict,writer,title,iteration_number):
    rows = 2
    cols = 2
    depth_mask = images_dict['optical_flow_mask'][0][
        0, ...]  # flow_mask is duplicated across the feature dimension
    depth_mask_float = depth_mask.type(torch.int).detach().cpu().numpy()
    f = plt.figure(figsize=(6, 4))
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    print('------------------------------------------------------------')
    print('------------------------------------------------------------')
    print('------------------------------------------------------------')
    source_depth_gt = images_dict['source_depth_gt'][0].permute(1, 2, 0).detach().cpu().numpy()
    target_depth_gt = images_dict['target_depth_gt'][0].permute(1, 2, 0).detach().cpu().numpy()
    source_depth_gt_Z = source_depth_gt[..., 2]
    target_depth_gt_Z = target_depth_gt[..., 2]
    source_depth_gt_masked = source_depth_gt_Z * depth_mask_float
    # max_z_value = max(source_depth_gt_masked.flatten())
    # min_z_value = min(source_depth_gt_viz.flatten())
    viz_max = 4
    print(f'source_depth_gt_masked - {np.mean(source_depth_gt_masked.flatten())}')
    a = f.add_subplot(rows, cols, 1)
    a.set_axis_off()
    a.set_title("source_depth_gt")
    plt.imshow(source_depth_gt_masked, cmap="jet")
    plt.clim(0, viz_max)
    a = f.add_subplot(rows, cols, 3)
    a.set_axis_off()
    a.set_title("target_depth_gt")
    plt.imshow(target_depth_gt_Z, cmap="jet")
    plt.clim(0, viz_max)
    depth_pred_source = images_dict['depth_pred_source'][0].squeeze().detach().cpu().numpy()
    depth_pred_target = images_dict['depth_pred_target'][0].squeeze().detach().cpu().numpy()
    depth_pred_source_masked = depth_pred_source * depth_mask_float
    print(f'depth_pred_source_masked - {np.mean(depth_pred_source_masked.flatten())}')
    a = f.add_subplot(rows, cols, 2)
    a.set_axis_off()
    a.set_title("depth_pred_source")
    plt.imshow(depth_pred_source_masked, cmap="jet")
    plt.clim(0, viz_max)

    a = f.add_subplot(rows, cols, 4)
    a.set_axis_off()
    a.set_title("depth_pred_target")
    plt.imshow(depth_pred_target, cmap="jet")
    plt.clim(0, viz_max)
    f.tight_layout()
    # plt.show()
    writer.add_figure(title + '_depth_prediction', f, iteration_number, close=True)


def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3

    T_opengl_cv = np.array(
        [[1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]]
    )

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


class CustomDrawGeometryWithKeyCallback():
    def __init__(self, geometry_dict, alignment_dict, corresp_set):
        self.added_source_pcd = True
        self.added_source_obj = False
        self.added_target_pcd = False
        self.added_graph = False

        self.added_both = False

        self.added_corresp = False
        self.added_weighted_corresp = False

        self.aligned = False

        self.rotating = False
        self.stop_rotating = False

        self.source_pcd = geometry_dict["source_pcd"]
        self.source_obj = geometry_dict["source_obj"]
        self.target_pcd = geometry_dict["target_pcd"]
        self.graph      = geometry_dict["graph"]

        # align source to target
        self.valid_source_points_cached = alignment_dict["valid_source_points"]
        self.valid_source_colors_cached = copy.deepcopy(self.source_obj.colors)
        self.line_segments_unit_cached  = alignment_dict["line_segments_unit"]
        self.line_lengths_cached        = alignment_dict["line_lengths"]

        self.good_matches_set          = corresp_set["good_matches_set"]
        self.good_weighted_matches_set = corresp_set["good_weighted_matches_set"]
        self.bad_matches_set           = corresp_set["bad_matches_set"]
        self.bad_weighted_matches_set  = corresp_set["bad_weighted_matches_set"]

        # correspondences lines
        self.corresp_set = corresp_set

    def remove_both_pcd_and_object(self, vis, ref):
        if ref == "source":
            if self.added_source_pcd:
                vis.remove_geometry(self.source_pcd)
                self.added_source_pcd = False

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False
        elif ref == "target":
            if self.added_target_pcd:
                vis.remove_geometry(self.target_pcd)
                self.added_target_pcd = False

    def clear_for(self, vis, ref):
        if self.added_both:
            if ref == "target":
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.added_both = False

    def get_name_of_object_to_record(self):
        if self.added_both:
            return "both"

        if self.added_graph:
            if self.added_source_pcd:
                return "source_pcd_with_graph"
            if self.added_source_obj:
                return "source_obj_with_graph"
        else:
            if self.added_source_pcd:
                return "source_pcd"
            if self.added_source_obj:
                return "source_obj"

        if self.added_target_pcd:
            return "target_pcd"

    def custom_draw_geometry_with_key_callback(self):

        def toggle_graph(vis):
            if self.graph is None:
                print("You didn't pass me a graph!")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_graph:
                for g in self.graph:
                    vis.remove_geometry(g)
                self.added_graph = False
            else:
                for g in self.graph:
                    vis.add_geometry(g)
                self.added_graph = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_obj(vis):
            print("::toggle_obj")

            if self.added_both:
                print("-- will not toggle obj. First, press either S or T, for source or target pcd")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            # Add source obj
            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False
            # Add source pcd
            elif self.added_source_obj:
                vis.add_geometry(self.source_pcd)
                vis.remove_geometry(self.source_obj)
                self.added_source_pcd = True
                self.added_source_obj = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_source(vis):
            print("::view_source")

            self.clear_for(vis, "source")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_source_pcd:
                vis.add_geometry(self.source_pcd)
                self.added_source_pcd = True

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.remove_both_pcd_and_object(vis, "target")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_target(vis):
            print("::view_target")

            self.clear_for(vis, "target")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.remove_both_pcd_and_object(vis, "source")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_both(vis):
            print("::view_both")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False

            if not self.added_source_obj:
                vis.add_geometry(self.source_obj)
                self.added_source_obj = True

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.added_both = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

        def rotate(vis):
            print("::rotate")

            self.rotating = True
            self.stop_rotating = False

            total = 2094
            speed = 5.0
            n = int(total / speed)
            for _ in range(n):
                ctr = vis.get_view_control()

                if self.stop_rotating:
                    return False

                ctr.rotate(speed, 0.0)
                vis.poll_events()
                vis.update_renderer()

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def rotate_slightly_left_and_right(vis):
            print("::rotate_slightly_left_and_right")

            self.rotating = True
            self.stop_rotating = False

            if self.added_both and not self.aligned:
                moves = ['lo', 'lo', 'pitch_f', 'pitch_b', 'ri', 'ro']
                totals = [(2094/4)/2, (2094/4)/2, 2094/4, 2094/4, 2094/4, 2094/4]
                abs_speed = 5.0
                abs_zoom = 0.15
                abs_pitch = 5.0
                iters_to_move = [range(int(t / abs_speed)) for t in totals]
                stop_at = [True, True, True, True, False, False]
            else:
                moves = ['ro', 'li']
                total = 2094 / 4
                abs_speed = 5.0
                abs_zoom = 0.03
                iters_to_move = [range(int(total / abs_speed)), range(int(total / abs_speed))]
                stop_at = [True, False]

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom / 4.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        toggle_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    if move_idx == 1:
                        toggle_weighted_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    time.sleep(0.5)

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def align(vis):
            if not self.added_both:
                return False

            moves = ['lo', 'li', 'ro']
            totals = [(2094/4)/2, (2094/4), 2094/4 + (2094/4)/2]
            abs_speed = 5.0
            abs_zoom = 0.15
            abs_pitch = 5.0
            iters_to_move = [range(int(t / abs_speed)) for t in totals]
            stop_at = [True, True, True]

            # Paint source object yellow, to differentiate target and source better
            # self.source_obj.paint_uniform_color([0, 0.8, 0.506])
            self.source_obj.paint_uniform_color([1, 0.706, 0])
            vis.update_geometry(self.source_obj)
            vis.poll_events()
            vis.update_renderer()

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom/50.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom/5.0

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        n_iter = 125
                        for align_iter in range(n_iter+1):
                            p = float(align_iter) / n_iter
                            self.source_obj.points = o3d.utility.Vector3dVector( self.valid_source_points_cached + self.line_segments_unit_cached * self.line_lengths_cached * p )
                            vis.update_geometry(self.source_obj)
                            vis.poll_events()
                            vis.update_renderer()

                    time.sleep(0.5)

            self.aligned = True

            return False

        def toggle_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False
            else:
                vis.add_geometry(self.good_matches_set)
                vis.add_geometry(self.bad_matches_set)
                self.added_corresp = True

            # also remove weighted corresp
            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_weighted_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False
            else:
                vis.add_geometry(self.good_weighted_matches_set)
                vis.add_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = True

            # also remove (unweighted) corresp
            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def reload_source_object(vis):
            self.source_obj.points = o3d.utility.Vector3dVector(self.valid_source_points_cached)
            self.source_obj.colors = self.valid_source_colors_cached

            if self.added_both:
                vis.update_geometry(self.source_obj)
                vis.poll_events()
                vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("G")] = toggle_graph
        key_to_callback[ord("S")] = view_source
        key_to_callback[ord("T")] = view_target
        key_to_callback[ord("O")] = toggle_obj
        key_to_callback[ord("B")] = view_both
        key_to_callback[ord("C")] = toggle_correspondences
        key_to_callback[ord("W")] = toggle_weighted_correspondences
        key_to_callback[ord(",")] = rotate
        key_to_callback[ord(";")] = rotate_slightly_left_and_right
        key_to_callback[ord("A")] = align
        key_to_callback[ord("Z")] = reload_source_object

        o3d.visualization.draw_geometries_with_key_callbacks([self.source_pcd], key_to_callback)


def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh
