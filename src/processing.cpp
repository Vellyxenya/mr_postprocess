#include "open3d/Open3D.h"
#include "processing.h"
#include "utils.h"
#include <iostream>
#include <tuple>
#include <igl/slice.h>

using namespace open3d;

using std::cout;
using std::endl;

int minimum_children = 10;
std::vector<geometry::PointCloud> transformed_pcds; //global since we want to reuse it for testing repeatedly the global optimization

Eigen::Matrix4d get_transformation_(const geometry::PointCloud& source,
    const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat, 
    const float max_correspondance_dist,
    const double kernel_param) {

    int nb_iterations = 300;

    double voxel_size = 0.02;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 4; ++i) {
        auto source_down = source.VoxelDownSample(voxel_size);
        source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto target_down = target.VoxelDownSample(voxel_size);
        target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto loss = pipelines::registration::TukeyLoss(kernel_param);
        auto kernel = loss.k_;
        auto result = pipelines::registration::RegistrationGeneralizedICP(
            *source_down, *target_down, max_correspondance_dist, T,
            pipelines::registration::TransformationEstimationForGeneralizedICP(kernel),
            pipelines::registration::ICPConvergenceCriteria(1e-7, 1e-7, nb_iterations));
        T = result.transformation_;
        voxel_size /= 2;
    }
    InfoMat = pipelines::registration::GetInformationMatrixFromPointClouds(source, target, max_correspondance_dist, T);
    return T;
}

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
PreprocessPointCloud(const geometry::PointCloud& pcd, const float voxel_size) {
    auto pcd_down = pcd.VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd_down, pcd_fpfh);
}

Eigen::Matrix4d global_registration(const geometry::PointCloud& source,
    const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat, 
    const float max_correspondance_dist,
    const double voxel_size) {
    std::shared_ptr<geometry::PointCloud> source_down, target_down;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    std::tie(source_down, source_fpfh) = PreprocessPointCloud(source, voxel_size);
    std::tie(target_down, target_fpfh) = PreprocessPointCloud(target, voxel_size);

    pipelines::registration::RegistrationResult registration_result;

    // Prepare checkers
    std::vector<std::reference_wrapper<
            const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
    auto correspondence_checker_edge_length =
            pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
    auto correspondence_checker_distance =
            pipelines::registration::CorrespondenceCheckerBasedOnDistance(max_correspondance_dist);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);

    bool mutual_filter = false;
    float confidence = 0.999;
    int max_iterations = 4000;
    registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
        *source_down, *target_down, *source_fpfh, *target_fpfh,
        mutual_filter, max_correspondance_dist,
        pipelines::registration::TransformationEstimationPointToPoint(false),
        3, correspondence_checker,
        pipelines::registration::RANSACConvergenceCriteria(max_iterations, confidence));
    return registration_result.transformation_;
}

// Eigen::Matrix4d get_transformation(const PCD& source_pcd,
//     const PCD& target_pcd, Eigen::Matrix6d& InfoMat, float max_correspondance_dist) {

//     const geometry::PointCloud source = geometry::PointCloud(source_pcd);
//     const geometry::PointCloud target = geometry::PointCloud(target_pcd);

//     return get_transformation_(source, target, InfoMat, max_correspondance_dist);
// }

void run_global_optimization(const std::vector<geometry::PointCloud>& partial_pcds, Eigen::MatrixXd& ThePointCloud,
    float global_opti_max_correspondance_dist, float edge_certainty, 
    double edge_prune_threshold, double preference_loop_closure, double feature_voxel_size) {
    //Setup the pose graph
    pipelines::registration::PoseGraph pose_graph;
    for(int src_id = 0; src_id < partial_pcds.size(); src_id++) {
        std::cout << "Computing transformations for point cloud " << src_id << std::endl;
        pose_graph.nodes_.push_back(
            pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Identity()));
        for(int tgt_id = src_id + 1; tgt_id < partial_pcds.size(); tgt_id++) {
            Eigen::Matrix6d InfoMat;
            Eigen::Matrix4d T = get_transformation_(partial_pcds[src_id], partial_pcds[tgt_id], 
               InfoMat, global_opti_max_correspondance_dist, feature_voxel_size);
            // Eigen::Matrix4d T = global_registration(partial_pcds[src_id], partial_pcds[tgt_id], 
            //     InfoMat, global_opti_max_correspondance_dist, feature_voxel_size);
            cout << "T:\n" << T << endl;
            bool uncertain = !(tgt_id == src_id + 1);
            float e_c = (tgt_id == src_id + 1)? 1 : 0.3;
            pose_graph.edges_.push_back(
                pipelines::registration::PoseGraphEdge(src_id, tgt_id, T, InfoMat, uncertain, e_c));
        }
    }
    //Setup optimization parameters
    pipelines::registration::GlobalOptimizationLevenbergMarquardt optimization_method;
    pipelines::registration::GlobalOptimizationConvergenceCriteria criteria; //TODO probably need to tweak
    int reference_node = 0;
    auto option = pipelines::registration::GlobalOptimizationOption(
        global_opti_max_correspondance_dist, edge_prune_threshold, preference_loop_closure, reference_node);

    //Run global optimization
    std::cout << "Running Global optimization..." << std::endl;
    pipelines::registration::GlobalOptimization(pose_graph, optimization_method, criteria, option);

    std::cout << "Merging Into global point cloud..." << std::endl;
    geometry::PointCloud pcd_combined;
    for(int i = 0; i < partial_pcds.size(); i++) {
        std::cout << '.' << std::flush;
        Eigen::Matrix4d Transformation = pose_graph.nodes_[i].pose_;
        cout << "Transformation:\n" << Transformation << endl;
        geometry::PointCloud transformed_pcd = geometry::PointCloud(partial_pcds[i]); //Copy because Transform is non-const
        pcd_combined += transformed_pcd.Transform(Transformation);
    }
    cout << "Combined " << pcd_combined.points_.size() << endl;
    std::shared_ptr<geometry::PointCloud> pcd_combined_down = pcd_combined.VoxelDownSample(0.005);
    ThePointCloud = vec_to_eigen(pcd_combined_down->points_);
}

void rerun_global_optimization(Eigen::MatrixXd& ThePointCloud, float global_opti_max_correspondance_dist,
    float edge_certainty, double edge_prune_threshold, double preference_loop_closure, double feature_voxel_size) {

    run_global_optimization(transformed_pcds, ThePointCloud, 
        global_opti_max_correspondance_dist, edge_certainty, 
        edge_prune_threshold, preference_loop_closure, feature_voxel_size);
}

void filter_and_save_partial_pcd(const geometry::PointCloud& pcd_combined, 
    const Eigen::Matrix4d& odometry, std::vector<Eigen::MatrixXd>& partial_pcds, 
    std::vector<Eigen::Matrix4d>& partial_odometries) {

    std::shared_ptr<open3d::geometry::PointCloud> pc;
    std::vector<size_t> indices;
    bool statistical_removal = true;
    if(statistical_removal) {
        std::tie(pc, indices) = pcd_combined.RemoveStatisticalOutliers(50, 1);
    } else {
        std::tie(pc, indices) = pcd_combined.RemoveRadiusOutliers(20, 0.01);
    }
    Eigen::MatrixXd Points = vec_to_eigen(pcd_combined.points_);
    Eigen::MatrixXd FilteredPoints(indices.size(), 3);
    cout << FilteredPoints.rows() << "/" << Points.rows() << endl;
    int k = 0;
    for(int i = 0; i < indices.size(); i++) {
        FilteredPoints.row(k++) = Points.row(indices[i]);
    }
    partial_pcds.push_back(FilteredPoints);
    partial_odometries.push_back(odometry);
}

std::shared_ptr<geometry::TriangleMesh> poisson_reconstruction(geometry::PointCloud& point_cloud) {
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<double> densities;
    point_cloud.EstimateNormals();
    float scale = 3;
    std::tie(mesh, densities) = geometry::TriangleMesh::CreateFromPointCloudPoisson(point_cloud, 8UL, 0, scale);
    io::WriteTriangleMesh("../out/FinalMesh.ply", *mesh);
    return mesh;
}

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec, 
    std::vector<PCD>& colors_vec, 
    std::vector<Eigen::MatrixXd>& list_cumulative_pcds,
    std::vector<Eigen::MatrixXd>& partial_pcds,
    Eigen::MatrixXd& ThePointCloud,
    float global_opti_max_correspondance_dist, float edge_certainty, 
    double edge_prune_threshold, double preference_loop_closure, double kernel_param) {

    double max_correspondance_dist = 0.01;
    Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
    geometry::PointCloud pcd_combined;
    std::vector<Eigen::Matrix4d> odometries;
    odometries.push_back(odometry);

    std::vector<geometry::PointCloud> pcds;
    for(int i = 0; i < pcds_vec.size(); i++) {
        auto pcd = geometry::PointCloud(pcds_vec[i]);
        pcd.colors_ = colors_vec[i];
        pcds.push_back(pcd);
    }

    std::vector<Eigen::Matrix4d> partial_odometries;
    std::vector<Eigen::Matrix4d> partial_Ts;
    int nb_children_pcds = 1;
    for(int src_id = 0; src_id < pcds.size()-1; src_id++) {
        Eigen::Matrix6d InfoMat;
        Eigen::Matrix4d T = get_transformation_(pcds[src_id], pcds[src_id+1], InfoMat, max_correspondance_dist, kernel_param);
        Eigen::Matrix4d new_odometry = T * odometry;
        double norm = (new_odometry - odometry).squaredNorm();
        odometry = new_odometry;
        //std::cout << "norm: " << norm << std::endl;

        float error_threshold = 0.004;
        if(norm > error_threshold) {
            pcd_combined = pcds[src_id];
            std::cout << nb_children_pcds << std::endl;
            if(nb_children_pcds >= minimum_children) {
                filter_and_save_partial_pcd(pcd_combined, odometry, partial_pcds, partial_odometries);
            }
            nb_children_pcds = 1;
            continue;
        }

        Eigen::Matrix4d inv_odo = odometry.inverse();
        pcd_combined += pcds[src_id].Transform(inv_odo);
        pcd_combined = *pcd_combined.VoxelDownSample(0.01);
        list_cumulative_pcds.push_back(vec_to_eigen(pcd_combined.points_));
        nb_children_pcds++;
    }
    if(nb_children_pcds >= minimum_children) {
        filter_and_save_partial_pcd(pcd_combined, odometry, partial_pcds, partial_odometries);
    }

    //TODO: Clean every point cloud
    cout << "Putting all clouds in the same frame..." << endl;
    geometry::PointCloud final_pcd_combined;
    int nb_merges = partial_pcds.size();
    for(int i = 0; i < nb_merges; i++) {
        cout << '.' << std::flush;
        Eigen::Matrix4d Transformation = partial_odometries[i].inverse();
        auto transformed_pc = geometry::PointCloud(eigen_to_vec(partial_pcds[i])).Transform(Transformation);
        transformed_pcds.push_back(transformed_pc);
        final_pcd_combined += transformed_pc;
        cout << "final_pcd_combined.size(): " << final_pcd_combined.points_.size() << endl;
    }
    cout << "\nFinished merging" << endl;
    Eigen::MatrixXd MergedPoints = vec_to_eigen(final_pcd_combined.points_);
    partial_pcds.push_back(MergedPoints); //TODO put in separate datastructure

    run_global_optimization(transformed_pcds, ThePointCloud,
        global_opti_max_correspondance_dist, edge_certainty, 
        edge_prune_threshold, preference_loop_closure, kernel_param);

    auto ThePointCloud_o3d = geometry::PointCloud((eigen_to_vec(ThePointCloud)));
    auto mesh = poisson_reconstruction(ThePointCloud_o3d);
    // mesh->vertices_;
    // mesh->triangles;

    return list_cumulative_pcds[list_cumulative_pcds.size()-1];
}