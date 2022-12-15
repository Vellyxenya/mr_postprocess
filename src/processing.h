#ifndef PROCESSING_H
#define PROCESSING_H

typedef std::vector<Eigen::Vector3d> PCD;

Eigen::Matrix4d get_transformation(const PCD& source_pcd,
    const PCD& target_pcd, Eigen::Matrix<double, 6, 6>& InfoMat, float max_correspondance_dist);

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec, 
    std::vector<PCD>& colors_vec, 
    std::vector<Eigen::MatrixXd>& list_cumulative_pcds,
    std::vector<Eigen::MatrixXd>& partial_pcds,
    Eigen::MatrixXd& ThePointCloud,
    float global_opti_max_correspondance_dist, float edge_certainty, 
    double edge_prune_threshold, double preference_loop_closure, double kernel_param);

void rerun_global_optimization(Eigen::MatrixXd& ThePointCloud, float global_opti_max_correspondance_dist,
    float edge_certainty, double edge_prune_threshold, double preference_loop_closure, double feature_voxel_size);

#endif