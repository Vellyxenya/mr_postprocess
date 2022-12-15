#include "loading.h"

#include <filesystem>
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <iostream>
#include "utils.h"

using std::vector;

typedef std::vector<std::vector<unsigned short int>> DepthImage;
typedef vector<vector<Eigen::RowVector3d>> RGBImage;

int read_paths(std::string folder, std::vector<std::string>& paths) {
  for (const auto & entry : std::filesystem::directory_iterator(folder+"Depth AHaT")) {
    std::string path = entry.path();
    if(path[path.length()-7] == '_') //discard _ab.pgm files
      continue;
    paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());
  return paths.size();
}

DepthImage read_pgm(std::string pgm_file_path) {  
  int row = 0, col = 0, num_of_rows = 0, num_of_cols = 0;
  std::stringstream ss;    
  std::ifstream infile(pgm_file_path, std::ios::binary);

  std::string inputLine = "";
  std::getline(infile, inputLine); //read the first line : P5
  if(inputLine.compare("P5") != 0) std::cerr << "Version error" << std::endl;

  ss << infile.rdbuf(); //read the third line : width and height
  ss >> num_of_cols >> num_of_rows;

  int max_val; //maximum intensity value
  ss >> max_val;
  ss.ignore();

  uint16_t pixel;
  DepthImage data(num_of_rows, std::vector<uint16_t>(num_of_cols));
  for (row = 0; row < num_of_rows; row++) {
    for (col = 0; col < num_of_cols; col++) {
      ss.read((char*)&pixel, 2);
      endian_swap(pixel);
      data[row][col] = pixel;
    }
  }
  return data;
}

RGBImage read_rgb(std::string folder, long pv_timestamp, int width, int height) {
  std::string file_path = folder+"PV/"+std::to_string(pv_timestamp)+".bytes";
  std::ifstream infile(file_path, std::ios::binary);
  unsigned int val;
  int y = 0;
  int x = 0;
  Eigen::MatrixXf RGB;
  RGBImage colors = RGBImage(height, vector<Eigen::RowVector3d>(width));
  RGB.resize(height, width);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      infile.read(reinterpret_cast<char*>(&val), sizeof(unsigned int));
      int r = (val & 0x00FF0000) >> 16;
      int g = (val & 0x0000FF00) >> 8;
      int b = (val & 0x000000FF);
      colors[y][x] = Eigen::RowVector3d(r, g, b) / 255.0;
    }
  }
  return colors;
}

void read_extrinsics(std::string folder, Eigen::MatrixXf& Ext) {
  std::ifstream infile(folder+"Depth AHaT_extrinsics.txt");
  std::string inputLine = "";
  Ext.resize(4, 4);
  int j = 0, i = 0;
  while(std::getline(infile, inputLine, ',')){
    float f = std::stof(inputLine);
    Ext(j, i) = f;
    i++;
    if(i == 4) {
      i = 0;
      j++;
    }
  }
}

void read_lut(std::string folder, Eigen::MatrixXf& mat) {
  std::ifstream infile(folder+"Depth AHaT_lut.bin", std::ios::binary);
  float f;
  int y = 0;
  int x = 0;
  mat.resize(512 * 512, 3);
  while (infile.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    mat(y, x) = f;
    x++;
    if(x >= 3) {
      x = 0;
      y++;
    }
  }
}

void read_rig2world(std::string folder, std::vector<Eigen::MatrixXf>& rig2world_vec, std::vector<long>& timestamps) {
  std::ifstream infile(folder+"Depth AHaT_rig2world.txt");
  std::string input_line = "";
  while(getline(infile, input_line)) {
    Eigen::MatrixXf M(4, 4);
    int j = 0, i = 0;
    std::stringstream line(input_line);
    std::string timestamp;
    std::string next_val;
    getline(line, timestamp, ',');
    timestamps.push_back(stol(timestamp));
    while(getline(line, next_val, ',')){
      float f = std::stof(next_val);
      M(j, i) = f;
      i++;
      if(i == 4) {
        i = 0;
        j++;
      }
    }
    rig2world_vec.push_back(M);
  }
}

void read_pv_meta(std::string folder, std::vector<Eigen::MatrixXf>& pv2world_matrices, std::vector<long>& timestamps, 
  std::vector<std::pair<float, float>>& focals, float& intrinsics_ox, float& intrinsics_oy,
  int& intrinsics_width, int& intrinsics_height) {
  
  std::ifstream infile(find_file_ending_with(folder, "_pv.txt"));
  std::string input_line = "";

  getline(infile, input_line, ',');
  intrinsics_ox = stof(input_line);
  getline(infile, input_line, ',');
  intrinsics_oy = stof(input_line);
  getline(infile, input_line, ',');
  intrinsics_width = stof(input_line);
  getline(infile, input_line);
  intrinsics_height = stof(input_line);

  while(true) {
    if(!getline(infile, input_line)) break;
    Eigen::MatrixXf M(4, 4);
    
    std::stringstream line(input_line);
    std::string next_val;
    getline(line, next_val, ',');
    timestamps.push_back(stol(next_val));
    getline(line, next_val, ',');
    int focalx = stof(next_val);
    getline(line, next_val, ',');
    int focaly = stof(next_val);

    focals.push_back(std::pair<float, float>(focalx, focaly));

    for(int j = 0; j < 4; j++) {
      for(int i = 0; i < 4; i++) {
        getline(line, next_val, ',');
        M(j, i) = std::stof(next_val);
      }
    }
    pv2world_matrices.push_back(M);
  }
}

void read_human_data(std::string folder, vector<long>& timestamps, 
  vector<vector<Eigen::MatrixXf>>& list_joints_left,
  vector<vector<Eigen::MatrixXf>>& list_joints_right,
  vector<Eigen::VectorXf>& list_gaze_origins,
  vector<Eigen::VectorXf>& list_gaze_directions,
  vector<float>& list_gaze_distances,
  vector<Eigen::MatrixXf>& list_head_data) {

  std::ifstream infile(find_file_ending_with(folder, ".csv"));
  std::string input_line = "";
  const int joint_count = 26;

  while(true) {
    if(!getline(infile, input_line)) break;
    std::stringstream line(input_line);
    std::string next_val;

    //Read timestamp
    getline(line, next_val, ',');
    long timestamp = stol(next_val);

    //Read head data
    Eigen::MatrixXf Head(4, 4);
    for(int j = 0; j < 4; j++) {
      for(int i = 0; i < 4; i++) {
        getline(line, next_val, ',');
        Head(j, i) = std::stof(next_val);
      }
    }
    list_head_data.push_back(Head);
    
    //Read left hand data
    getline(line, next_val, ',');
    bool left_hand_available = std::stoi(next_val) == 1;
    vector<Eigen::MatrixXf> joints_left;
    for(int joint = 0; joint < joint_count; joint++) {
      Eigen::MatrixXf Joint(4, 4);
      for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
          getline(line, next_val, ',');
          Joint(j, i) = std::stof(next_val);
        }
      }
      joints_left.push_back(Joint);
    }

    //Read right hand data
    getline(line, next_val, ',');
    bool right_hand_available = std::stoi(next_val) == 1;
    vector<Eigen::MatrixXf> joints_right;
    for(int joint = 0; joint < joint_count; joint++) {
      Eigen::MatrixXf Joint(4, 4);
      for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
          getline(line, next_val, ',');
          Joint(j, i) = std::stof(next_val);
        }
      }
      joints_right.push_back(Joint);
    }

    //Only push timestamp if both left and right hand data are available
    if(left_hand_available && right_hand_available) {
      timestamps.push_back(timestamp);
      list_joints_left.push_back(joints_left);
      list_joints_right.push_back(joints_right);
    }

    //Read gaze data
    getline(line, next_val, ',');
    bool gaze_available = std::stoi(next_val) == 1;
    Eigen::VectorXf GazeOrigin(4);
    for(int i = 0; i < 4; i++) {
      getline(line, next_val, ',');
      GazeOrigin(i) = std::stof(next_val);
    }
    list_gaze_origins.push_back(GazeOrigin);

    Eigen::VectorXf GazeDirection(4);
    for(int i = 0; i < 4; i++) {
      getline(line, next_val, ',');
      GazeDirection(i) = std::stof(next_val);
    }
    list_gaze_directions.push_back(GazeDirection);

    getline(line, next_val, ',');
    float gaze_distance = std::stof(next_val);
    list_gaze_distances.push_back(gaze_distance);
  }
}