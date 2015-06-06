#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <stdio.h>

using namespace cv;
using namespace std;

static string tags[] = {"fa", "fb", "ql", "qr"};

/**
 * 이미지 1장 읽음.
 */
static void read_images(vector<Mat> &images, const string& dir, const string& tag, const string& id) {
	string path = dir+tag+"/"+id+"_"+tag+".jpg";
	images.push_back(imread(path, 0));
}

/**
 * id_list.csv 파일읽음.
 */
static void read_csv(const string& filename, const string& image_dir, vector<Mat>& images_fa, vector<Mat>& images_fb,
		vector<Mat>& images_ql, vector<Mat>& images_qr, vector<int>& ids) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		cerr << error_message << endl;
		exit(1);
	}
	string line, idstr;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, idstr);
		if (!idstr.empty()) {
			read_images(images_fa, image_dir, tags[0], idstr);
			read_images(images_fb, image_dir, tags[1], idstr);
			read_images(images_ql, image_dir, tags[2], idstr);
			read_images(images_qr, image_dir, tags[3], idstr);
			ids.push_back(atoi(idstr.c_str()));
		}
	}
}

/**
 * Eigenfaces 알고리즘으로 gallery - probe 유사도행렬 계산.
 */
static Mat_<double> calcSimilarity(Ptr<FaceRecognizer> model, int dims, vector<Mat> gallery, vector<Mat> probe, double& minutues) {
	Mat mean = model->getMat("mean");
	Mat eigenfaces = model->getMat("eigenvectors");

	// cut to 10 dims
	Mat evs = Mat(eigenfaces, Range::all(), Range(0, dims));

	cout << "calculating similarities.. dims(" << dims << ")" << endl;
	clock_t Start = clock();

	int len_g = gallery.size(), len_p = probe.size();
	Mat gi, pj;
	Mat sim = Mat_<double>(len_g,len_p);
	for (int i=0; i<len_g; i++) {
		gi = subspaceProject(evs, mean, gallery[i].reshape(1, 1));
		for (int j=0; j<len_p; j++) {
			pj = subspaceProject(evs, mean, probe[j].reshape(1, 1));
			double dist = norm(gi, pj, NORM_L2);
			sim.at<double>(i, j) = -1*dist;	// inverse value
		}
		cout << ">";
		cout.flush();
	}
	cout << endl;
	minutues = (clock() - Start)/(double)(1000*1000*60);
	return sim;
}

/**
 * CMC Curve를 그리기 위한 matching score 출력
 */
static void calcMathingScore(const char* filename, Mat_<double> sim) {
	cout << "calculating matching score.." << endl;
	ofstream cmc_file;
	cmc_file.open(filename);
	int cnt_r;
	for (int r=1; r<=100; r++) {
		cnt_r = 0;
		// calculate matching score
		for (int j=0; j<sim.cols; j++) {
			// for each pj in probe
			double s = sim.at<double>(j, j);
			// cout # of gk: s_kj >= s_jj
			int rank_pj = 0;
			for (int k=0; k<sim.rows; k++) {
				if (sim.at<double>(k, j) >= s)
					rank_pj++;
			}
			if (rank_pj <= r)
				cnt_r++;
		}
		cmc_file << r << "," << (double)cnt_r*100/(double)sim.cols << endl;
	}
	cmc_file.close();
}

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 3) {
		cout << "usage: " << argv[0] << " <csv file> <image parent folder> " << endl;
		exit(1);
	}

	// csv file
	string fn_csv = string(argv[1]);
	string image_dir = string(argv[2]);
	vector<Mat> images_fa;
	vector<Mat> images_fb;
	vector<Mat> images_ql;
	vector<Mat> images_qr;
	vector<int> ids;

	try {
		read_csv(fn_csv, image_dir, images_fa, images_fb, images_ql, images_qr, ids);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	if (!images_fa.size() || !images_fb.size() || !images_ql.size() || !images_qr.size()) {
		string error_message = "some data sets are empty!";
		cerr << error_message << endl;
		exit(1);
	}

	// calculate - gallery(fa data set) : prob(fb data set)
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	cout << "training gallery..." << endl;
	model->train(images_fa, ids);

	double minutes = 0;
	cout << "task: fa -> fb" << endl;
	// fa - fb
	char sbuf[100];
	ofstream time_file;
	string time_filenm = "/home/aaron/outputs/fa_fb/time.txt";
	time_file.open(time_filenm.c_str());
	for (int dims=10; dims<=100; dims+=10) {
		ostringstream ss;
		ss << "/home/aaron/outputs/fa_fb/cmc_fa_fb_" << dims << ".csv";
		Mat sim = calcSimilarity(model, dims, images_fa, images_fb, minutes);
		calcMathingScore(ss.str().c_str(), sim);

		sprintf(sbuf, "dims(%d): Takes: %.2f mins.\n", dims, minutes);
		time_file << sbuf << endl;
	}
	time_file.close();

	cout << "task: fa -> ql" << endl;
	// fa - ql
	time_filenm = "/home/aaron/outputs/fa_ql/time.txt";
	time_file.open(time_filenm.c_str());
	for (int dims=10; dims<=100; dims+=10) {
		ostringstream ss;
		ss << "/home/aaron/outputs/fa_ql/cmc_fa_ql_" << dims << ".csv";
		Mat sim = calcSimilarity(model, dims, images_fa, images_ql, minutes);
		calcMathingScore(ss.str().c_str(), sim);

		sprintf(sbuf, "dims(%d): Takes: %.2f mins.\n", dims, minutes);
		time_file << sbuf << endl;
	}
	time_file.close();

	cout << "task: fa -> qr" << endl;
	// fa - qr
	time_filenm = "/home/aaron/outputs/fa_qr/time.txt";
	time_file.open(time_filenm.c_str());
	for (int dims=10; dims<=100; dims+=10) {
		ostringstream ss;
		ss << "/home/aaron/outputs/fa_qr/cmc_fa_qr_" << dims << ".csv";
		Mat sim = calcSimilarity(model, dims, images_fa, images_qr, minutes);
		calcMathingScore(ss.str().c_str(), sim);

		sprintf(sbuf, "dims(%d): Takes: %.2f mins.\n", dims, minutes);
		time_file << sbuf << endl;
	}
	time_file.close();

	return 0;
}
