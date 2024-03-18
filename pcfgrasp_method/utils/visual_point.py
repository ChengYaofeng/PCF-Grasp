import open3d as o3d

def vis_pc(pc):
    """
        pc: {numpy} N x 3
    """
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("Pointcloud",960,720)
    pointcloud = o3d.geometry.PointCloud()

    while True:
        pointcloud.clear()
        pointcloud.points = o3d.utility.Vector3dVector(pc)

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True
        
        vis.update_geometry(pointcloud)
        vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()

        break