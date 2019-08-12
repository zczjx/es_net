#include <time.h>
#include <pthread.h>
#include <unistd.h>

#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define MAX_NR_QFRAMES (3)

using namespace cv;
using namespace std;

static void *capture_thread(void *arg);
static void *face_mark_thread(void *path_of_model);
static void *deque_watermaskd(void *arg);


static void print_fps(const char *fsp_dsc, long *fps, 
	long *pre_time, long *curr_time);


static bool cap_run = true;
static bool face_mark_run = false;

static pthread_mutex_t flock = PTHREAD_MUTEX_INITIALIZER;
deque<cv::Mat *> vframe_que;
deque<cv::Mat *> face_marked_que;


int main( int argc, char** argv)
{
	int err = true;
	pthread_t cap_tid, face_mark_tid, watermaskd_tid;
	long fps = 0; 
	long pre_time = 0;
	long curr_time = 0;
	Mat *vframe = NULL;

	cap_run = true;
	err = pthread_create(&cap_tid, NULL, capture_thread, NULL);
	err = pthread_create(&watermaskd_tid, NULL, deque_watermaskd, NULL);
	face_mark_run = true;
	err = pthread_create(&face_mark_tid, NULL, face_mark_thread, argv[1]);
		
	
	namedWindow("window3", (WINDOW_AUTOSIZE));
	while(1)
	{
		if(face_marked_que.empty())
			continue;
		
		// pthread_mutex_lock(&flock);
		cout << "vframe_que entry cnt: " << vframe_que.size() << endl;
		// pthread_mutex_unlock(&flock);
		print_fps("disp fps is: ", &fps, &pre_time, &curr_time);
		vframe = face_marked_que.front();
		face_marked_que.pop_front();
		imshow("window3", *vframe);

		if(NULL != vframe)
		{
			delete vframe;
			vframe = NULL;
		}

		if(waitKey(27) >= 0)
			break;    
	}
	
	cap_run = false;
	face_mark_run = false;
	destroyWindow("window3");
	return 0; 
}



static void *capture_thread(void *arg)
{
	VideoCapture vcap;
	Mat *tmp_frame = NULL;
	long fps = 0; 
	long pre_time = 0;
	long curr_time = 0;

	vcap.open(0);
	
	if(!vcap.isOpened())
	{
		cerr << "cannot open camera!!" << endl;
		pthread_exit(0);
	}

	vcap.set(CAP_PROP_FRAME_WIDTH, 320);
	vcap.set(CAP_PROP_FRAME_HEIGHT, 240);
	cout << "default fps: " << vcap.get(CAP_PROP_FPS) << endl;

	while(true == cap_run)
	{	

		tmp_frame = new Mat();
		vcap >> (*tmp_frame);
		vframe_que.push_back(tmp_frame);
		// print_fps("cap fps is: ", &fps, &pre_time, &curr_time);
		usleep(1000 * 160);
	}
	
	vcap.release();
}

static void *deque_watermaskd(void *arg)
{
	Mat *tmp_frame = NULL;
	int i, delta = 0;

	while(true == cap_run)
	{	
		pthread_mutex_lock(&flock);
		if(vframe_que.size() > MAX_NR_QFRAMES)
		{
			delta = vframe_que.size() - MAX_NR_QFRAMES;

			for(i = 0; i < delta; i++)
			{
				tmp_frame = vframe_que.front();
				vframe_que.pop_front();
				delete tmp_frame;
				tmp_frame = NULL;
			}

		}
		pthread_mutex_unlock(&flock);
			
		usleep(1000 * 500);
	}
	
}


static void *face_mark_thread(void *path_of_model)
{
	Scalar colors[] = {
		Scalar(0, 0, 255),
 		Scalar(0, 128, 255),
		Scalar(0, 255, 255),
		Scalar(0, 255, 0)
	};
		
	double scale = 1.0;
	
	int i, err;
	vector<cv::Rect> regions;
	CascadeClassifier classifier((char *) path_of_model);
	Mat *orig_vframe = NULL;
	Mat *gray_img = NULL;
	// Mat *small_img = NULL;

	while(true == face_mark_run)
	{
		if(vframe_que.empty())
			continue;
		
		pthread_mutex_lock(&flock);
		orig_vframe = vframe_que.front();
		vframe_que.pop_front();
		pthread_mutex_unlock(&flock);
		
		gray_img = new Mat(orig_vframe->size(), CV_8UC1);
		cvtColor(*orig_vframe, *gray_img, COLOR_BGR2GRAY);
		equalizeHist(*gray_img, *gray_img);
		
		/**
		classifier.detectMultiScale(*gray_img, regions, 
					1.1, 3, 
					CV_HAAR_FIND_BIGGEST_OBJECT |
					CV_HAAR_SCALE_IMAGE |
					CV_HAAR_DO_CANNY_PRUNING,
					CV_HAAR_DO_ROUGH_SEARCH,
					Size(30, 30));
					**/
		
		i = 0;
		for(vector<cv::Rect>::iterator reg = regions.begin();
			reg != regions.end(); reg++, ++i)
		{
			Rect tmp = (*reg);
			tmp.x *= scale;
			tmp.y *= scale;
			tmp.width *= scale;
			tmp.height *= scale;
			rectangle(*orig_vframe, tmp, colors[i % 4], LINE_4);
		}
		face_marked_que.push_back(orig_vframe);

	}


}

static void print_fps(const char *fsp_dsc, long *fps, 
	long *pre_time, long *curr_time)
{
	struct timespec tp;

	clock_gettime(CLOCK_MONOTONIC, &tp);
	*curr_time = tp.tv_sec;
	(*fps)++;
		
	if(((*curr_time) - (*pre_time)) >= 1)
	{	
		cout << fsp_dsc << (*fps) << endl;
		*pre_time = *curr_time;
		*fps = 0;
	}
	
}


