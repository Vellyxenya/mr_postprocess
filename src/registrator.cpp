//
// Created by Noureddine Gueddach on 21/11/2022.
//

#include "open3d/Open3D.h"
#include "registrator.h"
#include <memory>
#include <chrono>
#include <string>
#include <fstream>
#include <queue>
#include <igl/copyleft/marching_cubes.h>

using namespace open3d;
using std::cout;
using std::endl;

std::unique_ptr<PCD> Registrator::getReconstructedPCD() const {
    PCD pcd;
    pcd.reserve(m_pcd->points_.size());
    for(const auto p : m_pcd->points_) {
        pcd.push_back(p);
    }
    return std::make_unique<PCD>(pcd);
}

Eigen::Matrix4d Registrator::getTransformation(const geometry::PointCloud& source,
    const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat, 
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
            *source_down, *target_down, m_max_corr_dist_transformation, T,
            pipelines::registration::TransformationEstimationForGeneralizedICP(kernel),
            pipelines::registration::ICPConvergenceCriteria(1e-7, 1e-7, nb_iterations));
        T = result.transformation_;
        voxel_size /= 2;
    }
    InfoMat = pipelines::registration::GetInformationMatrixFromPointClouds(source, target, 
        m_max_corr_dist_transformation, T);
    return T;
}

bool Registrator::isRegistrationSuccessful(const geometry::PointCloud& pcd, const Eigen::Matrix4d& T) const {
    auto result = pipelines::registration::EvaluateRegistration(pcd, *m_pcd, m_max_corr_dist_evaluation, T);
    auto correspondance_set = result.correspondence_set_;
    auto fitness = result.fitness_; //Corresponds to: correspondance_set.size() / pcd.points_.size()
    auto rmse = result.inlier_rmse_;
    //bool most_of_pcd_is_inlier = correspondance_set.size() >= 0.8 * pcd.points_.size(); //same as fitness
    cout << fitness << " " << rmse << " " << endl;
    bool high_fitness = fitness > m_min_fitness;
    bool low_rmse = rmse < m_max_rmse;
    return high_fitness && low_rmse;
}

void filter_pcd(geometry::PointCloud& pcd) {
    std::shared_ptr<open3d::geometry::PointCloud> pc;
    std::vector<size_t> indices;
    bool statistical_removal = false;
    if(statistical_removal) {
        std::tie(pc, indices) = pcd.RemoveStatisticalOutliers(20, 1);
    } else {
        std::tie(pc, indices) = pcd.RemoveRadiusOutliers(10, 0.015);
    }
    auto points = pcd.points_;
    auto filtered_points = PCD();
    filtered_points.reserve(indices.size());
    for(int i = 0; i < indices.size(); i++) {
        filtered_points.push_back(pcd.points_[indices[i]]);
    }
    pcd.points_ = filtered_points;
}

bool Registrator::mergePCD(const PCD& pcd_) {
    auto pcd = geometry::PointCloud(pcd_);
    if(m_pcd == nullptr) { //First registration is always successful as it initializes the point cloud
        m_pcd = std::make_shared<geometry::PointCloud>(pcd);
        return true;
    }

    //DBSCAN
    bool dbscan = false; //Note: DBSCAN is too slow for real-time (could be use) for a final pass though
    if(dbscan) {
        std::vector<int> indices = pcd.ClusterDBSCAN(0.1, 0.7 * pcd.points_.size());
        PCD valid_points;
        for(int i = 0; i < indices.size(); i++) {
            if(indices[i] != -1)
                valid_points.push_back(pcd.points_[i]);
        }
        pcd.points_ = valid_points;
    }
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    //Compute the transformation between the current and global point cloud
    Eigen::Matrix6d InfoMat;
    double kernel_param = 0.01;
    Eigen::Matrix4d T = getTransformation(pcd, *m_pcd, InfoMat, kernel_param);

    //Evaluate the registration
    bool success = isRegistrationSuccessful(pcd, T);
    //If not successful, keep the global point cloud as is, wait for the user to realign
    if (!success) return false;

    filter_pcd(pcd);

    *m_pcd = m_pcd->Transform(T.inverse()); //Bring the global point cloud into the reference of the current frame
    *m_pcd += pcd; //Merge the current frame to the global point cloud
    m_pcd = m_pcd->VoxelDownSample(0.005); //downsample for performance
    //filter_pcd(*m_pcd);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return true;
}

void Registrator::denoise(const std::shared_ptr<open3d::geometry::PointCloud>& pcd) const {
    try {
        std::string noisy_file_name = "noisy_pcd.xyz";
        std::string denoised_file_name = "denoised_pcd.xyz";
        cout << "\nWriting  noisy pcd to file: " + noisy_file_name << endl;
        std::ofstream ostream("../noisy/" + noisy_file_name, std::ofstream::out);
        if (ostream.is_open()) {
            for (int i = 0; i < pcd->points_.size(); i++) {
                ostream << pcd->points_[i](0) << " " << pcd->points_[i](1) << " " << pcd->points_[i](2) << "\n";
            }
            ostream.close();
        } else {
            cout << "Could not create file: " + noisy_file_name << endl;
        }
        cout << "Finished writing noisy pcd to file! Handing over to Python" << endl;
        std::string command = std::string("cd ../ext/score-denoise && python test_single.py --input_xyz ../../noisy/") 
            + noisy_file_name + " --output_xyz ../../denoised/" + denoised_file_name;
        system(command.c_str());
        cout << "Finished denoising!" << endl;

        //Read back the denoised point cloud
        cout << "C++ takes over. Reading the denoised point cloud..." << endl;
        pcd->points_.clear();
        //std::ifstream istream("../denoised/" + denoised_file_name, std::ifstream::in);
        std::ifstream istream("../finaldenoisedboxdb.xyz", std::ifstream::in);
        for(std::string line; std::getline(istream, line); ) { //read stream line by line
            std::istringstream in(line); //make a stream for the line itself
            float x, y, z;
            in >> x >> y >> z;
            pcd->points_.push_back(Eigen::Vector3d(x, y, z));
        }
        cout << "Finished reading back the denoised pcd" << endl;

    } catch (const char* msg) {
        std::cerr << msg << endl;
    }
}

void Registrator::flood(const std::shared_ptr<open3d::geometry::PointCloud>& pcd, 
    const Eigen::Vector3d& resolution, Eigen::MatrixXd& grid_points, Eigen::VectorXd& grid_values) const {
    using std::vector;
    //#########################################
    //Note: Assumes the point cloud is closed #
    //#########################################

    //Discretize points and create a thick boundary out of them
    Eigen::Vector3d min = pcd->GetMinBound();
    Eigen::Vector3d max = pcd->GetMaxBound();
    Eigen::Vector3d extents = max - min;
    Eigen::Vector3d extents_inv = Eigen::Vector3d(1/extents.x(), 1/extents.y(), 1/extents.z());
    size_t n = pcd->points_.size();
    Eigen::Vector3d resolution_inv = Eigen::Vector3d(1/resolution.x(), 1/resolution.y(), 1/resolution.z());
    vector<vector<vector<uint8_t>>>* occupancy = new vector<vector<vector<uint8_t>>>(resolution.x(), 
        vector<vector<uint8_t>>(resolution.y(), vector<uint8_t>(resolution.z(), 0)));
    for(int i = 0; i < n; i++) {
        Eigen::Vector3d point = (pcd->points_[i] - min)
            .cwiseProduct(extents_inv).cwiseProduct(resolution);
        int x_ = (int)point.x();
        int y_ = (int)point.y();
        int z_ = (int)point.z();
        //Set each neighboring as a boundary point
        //This thickness is necessary so that we don't accidentally cross the boundary
        //when flooding
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                for(int k = -1; k <= 1; k++) {
                    int x = x_ + i;
                    int y = y_ + j;
                    int z = z_ + k;
                    if(x >= 0 && x < resolution.x() && 
                       y >= 0 && y < resolution.y() && 
                       z >= 0 && z < resolution.z()) {
                        (*occupancy)[x][y][z] = 1;
                    }
                }
            }
        }
    }
    
    //#######################################################################
    //Note: The key assumption here is that the center point will be inside #
    //the shape. If this does not hold then nothing works.                  #
    //#######################################################################

    //Compute the center of the point cloud.
    Eigen::Vector3d center = (pcd->GetCenter() - min)
        .cwiseProduct(extents_inv).cwiseProduct(resolution);
    std::queue<Eigen::Vector3i> q;
    q.push(Eigen::Vector3i((int)center.x(), (int)center.y(), (int)center.z()));
    vector<Eigen::Vector3d> inner_points;
    //Flood the inside of the shape, BFS style
    while(!q.empty()) {
        Eigen::Vector3i p = q.front();
        q.pop();
        if((*occupancy)[p.x()][p.y()][p.z()] != 0)
            continue;
        (*occupancy)[p.x()][p.y()][p.z()] = 2;
        Eigen::Vector3d w_p = Eigen::Vector3d(p.x(), p.y(), p.z())
            .cwiseProduct(resolution_inv).cwiseProduct(extents) + min;
        inner_points.push_back(w_p);
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                for(int k = -1; k <= 1; k++) {
                    int x = p.x() + i;
                    int y = p.y() + j;
                    int z = p.z() + k;
                    if(x >= 0 && x < resolution.x() && 
                       y >= 0 && y < resolution.y() && 
                       z >= 0 && z < resolution.z()) {
                        if(!(i == 0 && j == 0 && k == 0)) {
                            short val = (*occupancy)[x][y][z];
                            if(val == 0) {
                                q.push(Eigen::Vector3i(x, y, z));
                            }
                        }
                    }
                }
            }
        }
    }
    cout << "====SIZE: " << inner_points.size() << endl;

    //Expand the volume to compensate for the thick boundary
    vector<Eigen::Vector3i> expanded_points;
    for(int x = 0; x < resolution.x(); x++) {
        for(int y = 0; y < resolution.y(); y++) {
            for(int z = 0; z < resolution.z(); z++) {
                bool found_neighbor = false;
                if((*occupancy)[x][y][z] == 2)
                    continue;
                for(int i = -1; i <= 1 && !found_neighbor; i++) {
                    for(int j = -1; j <= 1 && !found_neighbor; j++) {
                        for(int k = -1; k <= 1 && !found_neighbor; k++) {
                            int x_ = x + i;
                            int y_ = y + j;
                            int z_ = z + k;
                            if(x_ >= 0 && x_ < resolution.x() && 
                               y_ >= 0 && y_ < resolution.y() && 
                               z_ >= 0 && z_ < resolution.z()) {
                                if((*occupancy)[x_][y_][z_] == 2) {
                                    expanded_points.push_back(Eigen::Vector3i(x, y, z));
                                    found_neighbor = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for(auto p : expanded_points) {
        (*occupancy)[p.x()][p.y()][p.z()] = 2;
    }

    //Put the result in the appropriate data structures in preparation for marching cubes
    //Also update the pcd, although from here on the pcd should not be relevant anymore
    int N = resolution.x() * resolution.y() * resolution.z();
    grid_points.resize(N, 3);
    grid_values.resize(N);
    grid_values.setOnes(); //points are outside the shape by default
    pcd->points_.clear();
    bool dense = true;
    int ii = 0;
    for(int x = 0; x < resolution.x(); x++) {
        for(int y = 0; y < resolution.y(); y++) {
            for(int z = 0; z < resolution.z(); z++, ii++) {
                Eigen::Vector3d w_p = Eigen::Vector3d(x, y, z)
                    .cwiseProduct(resolution_inv).cwiseProduct(extents) + min;
                grid_points.row(ii) = w_p.transpose();

                bool found_neighbor = false;
                if((*occupancy)[x][y][z] == 2) {
                    if(dense) {
                        pcd->points_.push_back(w_p);
                        grid_values(ii) = -2; //inner point
                    }
                    continue;
                }
                for(int i = -1; i <= 1 && !found_neighbor; i++) {
                    for(int j = -1; j <= 1 && !found_neighbor; j++) {
                        for(int k = -1; k <= 1 && !found_neighbor; k++) {
                            int x_ = x + i;
                            int y_ = y + j;
                            int z_ = z + k;
                            if(x_ >= 0 && x_ < resolution.x() && 
                               y_ >= 0 && y_ < resolution.y() && 
                               z_ >= 0 && z_ < resolution.z()) {
                                if((*occupancy)[x_][y_][z_] == 2) {
                                    found_neighbor = true;
                                }
                            }
                        }
                    }
                }
                if(found_neighbor) {
                    pcd->points_.push_back(w_p);
                    grid_values(ii) = -0.5; //inner point close to the surface
                }
            }
        }
    }
}

void Registrator::postProcess(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const {
    denoise(m_pcd);

    Eigen::Vector3d resolution = Eigen::Vector3d(70, 70, 70);
    Eigen::MatrixXd grid_points;
    Eigen::VectorXd grid_values;
    flood(m_pcd, resolution, grid_points, grid_values);

    igl::copyleft::marching_cubes(grid_values, grid_points, resolution.x(), resolution.y(), resolution.z(), V, F);
    m_pcd->points_.clear();
    for(int i = 0; i < V.rows(); i++) {
        m_pcd->points_.push_back(V.row(i).transpose());
    }
    cout << "Finished Marching Cubes" << endl;
}

void Registrator::saveReconstructedMesh(const std::string& save_path, Eigen::MatrixXd& V, Eigen::MatrixXi& F) const {
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<double> densities;

    denoise(m_pcd);

    Eigen::Vector3d resolution = Eigen::Vector3d(70, 70, 70);
    Eigen::MatrixXd grid_points;
    Eigen::VectorXd grid_values;
    flood(m_pcd, resolution, grid_points, grid_values);

    igl::copyleft::marching_cubes(grid_values, grid_points, resolution.x(), resolution.y(), resolution.z(), V, F);
    m_pcd->points_.clear();
    for(int i = 0; i < V.rows(); i++) {
        m_pcd->points_.push_back(V.row(i).transpose());
    }
    cout << "Finished Marching Cubes" << endl;

    // m_pcd->EstimateNormals();

    // float scale = 1;
    // std::tie(mesh, densities) = geometry::TriangleMesh::CreateFromPointCloudPoisson(*m_pcd, 9, 0, scale);
    // std::vector<bool> mask;
    // mask.reserve(densities.size());
    // for(int i = 0; i < densities.size(); i++) {
    //     //cout << densities[i] << " ";
    //     mask.push_back(densities[i] <= 2); //4.5);
    // }
    // //cout << endl;
    // mesh->RemoveVerticesByMask(mask);

    // std::vector<double> radii = {0.01, 0.005, 0.0025, 0.001};
    // mesh = geometry::TriangleMesh::CreateFromPointCloudBallPivoting(*m_pcd, radii);

    // double alpha = 0.02;
    // mesh = geometry::TriangleMesh::CreateFromPointCloudAlphaShape(*m_pcd, alpha);
    
    // io::WriteTriangleMesh(save_path, *mesh);
}