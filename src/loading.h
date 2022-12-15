#ifndef LOADING_H
#define LOADING_H

#include <vector>
#include <string>
#include <Eigen/Dense>

using std::vector;
using std::string;

typedef vector<vector<unsigned short int>> DepthImage;
typedef vector<vector<Eigen::RowVector3d>> RGBImage;

/***************************
******* DATA LOADING *******
****************************/

int read_paths(string folder, vector<string>& paths);

DepthImage read_pgm(string pgm_file_path);

RGBImage read_rgb(string folder, long pv_timestamp, int width, int height);

void read_extrinsics(string folder, Eigen::MatrixXf& Ext);

void read_lut(string folder, Eigen::MatrixXf& mat);

void read_rig2world(string folder, vector<Eigen::MatrixXf>& rig2world_vec, vector<long>& timestamps);

void read_pv_meta(string folder, vector<Eigen::MatrixXf>& pv2world_matrices, vector<long>& timestamps, 
  vector<std::pair<float, float>>& focals, float& intrinsics_ox, float& intrinsics_oy,
  int& intrinsics_width, int& intrinsics_height);

void read_human_data(string folder, vector<long>& timestamps, vector<vector<Eigen::MatrixXf>>& list_joints_left,
  vector<vector<Eigen::MatrixXf>>& list_joints_right,
  vector<Eigen::VectorXf>& list_gaze_origins,
  vector<Eigen::VectorXf>& list_gaze_directions,
  vector<float>& list_gaze_distances,
  vector<Eigen::MatrixXf>& list_head_data);

#endif