#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <sstream>

#define TRACE 1 // for trace fonction
#define DEBUG_MODE 0 //for debug trace
#define DEBUG if (DEBUG_MODE==1) // for debug trace

//f_picture_cropped size_of_train_picture image_name_prefix original_picture_new_size equalize_color
//prepare 0.3 100 p 800 1

#define PICTURE_CROPPED 0.3
#define TRAIN_SIZE 100
#define NEW_SIZE 800
#define EQUALIZE 1

// This values can be modify for your preference
#define SUCCESS 0
#define FAILURE -1

// Max of the images saved for future recognition
#define MAX_IMAGES 10

struct stat sb;

using namespace cv;
using namespace cv::face;
using namespace std;

// vars for paths of the images and CSV file
const string path_images = "../images/";
const string path_temp = "../temp/";
const string CSV_file = "../images/images.ext";

// vars for opencv path
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye.xml";
String glasses_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier glasses_cascade;

CvPoint Myeye_left;
CvPoint Myeye_right;

// when is detected one face, write in var if the fram contains eyes or glasses
string eyes_or_glasses = "";

/** Function Headers */
int detectAndDisplay(Mat frame);
int detectAndSaveFace(int faceId);
int detectAndReconizeFace(int faceId);
Mat rotate(Mat& image, double angle, CvPoint centre);
float Distance(CvPoint p1, CvPoint p2);
void trace(string s);
int CropFace(Mat &MyImage, CvPoint eye_left, CvPoint eye_right, CvPoint offset_pct,	CvPoint dest_sz);
void resizePicture(Mat& src, int coeff);
void readImageFolder(const String& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
int loadCascades();
void saveFaceFrame(Mat frame, int num_img, int label);
int  getLastLabel();
int prepareFaceImage(Mat &faceFrame);
int createCSV(int faceId);
int isFace(Mat framei, vector<Rect>& faces, vector<Rect>& eyes);

/*
 *
 * THE MAIN METHOD, YEAHH
 *
 * */
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Use: %s --help\n", argv[0]);
        return -1;
    }

    // verify correct usage for save_face
    if (argc != 3) {
        trace(format("(!!!)bad argument use: %s <command> <face_id>\n", argv[0]));
        return -1;
    }

    // Response of the function
    int response;
    // get faceId for identification
    int faceId = atoi(argv[2]);

    if (strcmp(argv[1], "save") == 0) {
        // prepare the folders for operation
        string command = format("rm %s*", path_temp.c_str());
        system(command.c_str());
        response = detectAndSaveFace(faceId);
    }
    else if (strcmp(argv[1], "recognize") == 0) {
        response = detectAndReconizeFace(faceId);
    }
    else {
        printf("Comand not found\n");
        return -1;
    }

    if (response == SUCCESS)
        printf("%s executed with succes\n", argv[1]);
    else
        printf("One error ocurred and %s not executed\n", argv[1]);


    return 0;
}

// Prepare frame for future face detection
int prepareFaceImage(Mat &faceFrame) {
    // verify de integrity of the frame
    if (faceFrame.empty()) {
        printf("Frame is NULL or EMPTY\n");
        return FAILURE;
    }

    // read parameters
    CvPoint Myoffset_pct;
    Myoffset_pct.x = 100.0 * PICTURE_CROPPED;
    Myoffset_pct.y = Myoffset_pct.x;

    // size of new picture
    CvPoint Mydest_sz;
    Mydest_sz.x = TRAIN_SIZE;
    Mydest_sz.y = Mydest_sz.x;

    // quality type JPG to save image
    vector<int> qualityType;
    qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
    qualityType.push_back(90);

    // new size of the picture
    int newSize = NEW_SIZE;
    if (faceFrame.size().width > newSize) {
        trace("(***)Image need to be resized\n");
        resizePicture(faceFrame, newSize);
    }

    if (detectAndDisplay(faceFrame) == SUCCESS) {
        if (CropFace(faceFrame, Myeye_left, Myeye_right, Myoffset_pct, Mydest_sz) == SUCCESS) {
            // conver to grayscale
            cvtColor(faceFrame, faceFrame, CV_BGR2GRAY);

            // equalyze histogram color
            if (EQUALIZE)
                equalizeHist(faceFrame, faceFrame);
        }

    } else
        return FAILURE;

    return SUCCESS;
}

// Detect 1 face and 2 eyes and save the image for database of the recognize
int detectAndSaveFace(int faceId) {
    // Load the cascades
    loadCascades();

    VideoCapture web_cam;
    Mat frame;
   // Read the video stream
    web_cam.open(-1);
    if (!web_cam.isOpened()) {
        printf("--(!)Error opening video capture\n");
        return FAILURE;
    }

    // Capture the last label insert in the base
    //int new_label = getLastLabel();

    // Number of the images save ofr recognition
    int imagesSaved = 0;
    while (web_cam.read(frame) && imagesSaved < MAX_IMAGES) {
        if (frame.empty()) {
            printf("--(!)No capture frame - Break!\n");
            break;
        }

        vector<Rect> faces;
        vector<Rect> eyes;

        // if frame is not a face with two eyes, then jump for the next interation of the loop
        if (isFace(frame, faces, eyes) == FAILURE)
            continue;

        Mat frame_gray;
        Rect face = faces[0];

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        // the original frame contains 1 face and 2 eyes
        Mat faceFrame = frame.clone();

        if (prepareFaceImage(faceFrame) == SUCCESS) {
            //saveFaceFrame(frame, num_img, new_label);
            string newName = format("%s%d.jpg", path_temp.c_str(), imagesSaved);
            imwrite(newName, faceFrame);
            trace(format("(***)image %s is saved\n", newName.c_str()));
            imagesSaved += 1;

            // Show green rectangle, when find 1 face and 2 eyes
            rectangle(frame, face, CV_RGB(0, 255, 0), 1);
        }
        
        imshow("FIND FACE", frame);

        // If press ESC remove all executed
        int c = waitKey(10);
        if ((char)c == 27) { break; }
    }

    // Verify the correct number of the images saved
    // then save the temp images in folder for future recognition
    if (imagesSaved == MAX_IMAGES)
        if (createCSV(faceId) == FAILURE)
            return FAILURE;

    return SUCCESS;
}

// Detect 1 face and 2 eyes, and explore images database for recognize the people
int detectAndReconizeFace(int faceId) {
    vector<Mat> images;
    vector<int> labels;

    cout << "call the readImagefolder" << endl;

    try {
        readImageFolder(CSV_file, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opennig file\"" << path_images << "\". Reason: " << e.msg << endl;
        return FAILURE;
    }

    // Get the height from de first image. We'll need this later in the code to reshape
    // the images to their original size AND we need to rechape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    
    cout << "width: " << im_width << " height: " << im_height << endl;

    /*
     * Mat test_sample = images[images.size() - 1];
    int test_label = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    */

    //Create the FaceRecognizer and trai it on the given images:
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);

    // load cascades
    loadCascades();

    VideoCapture web_cam;

    // Read the video stream
    web_cam.open(-1);
    if (!web_cam.isOpened()) {
        cerr << "Capture cannot be opened." << endl;
        return FAILURE;
    }

    // how many times, prediction is not eaull faceId
    int predictionNotEqual = 0;
    int predictionEqual = 0;
    // Holds the current frame for de video device
    Mat frame;
    while (web_cam.read(frame)) {
        // Clone the current frame
        Mat original = frame.clone();
        vector <Rect> faces;
        vector <Rect> eyes;

        // verify if the frame contains one face and two eyes
        // if no, jump for the next interaction of the loop
        if (isFace(frame, faces, eyes) == FAILURE)
            continue;

        Mat frame_gray;
        cvtColor(original, frame_gray, CV_BGR2GRAY);
        // Find faces in the frame
        //vector<Rect_<int>> faces;
        // At this point we have the position of the faces in faces.
        // Now we'll get the faces, make a prediction and annotate it in video.
        //for (int i = 0; i < faces.size(); i++) {
        Mat faceROI = frame_gray(faces[0]); // first frame
        
        // Process face by face:
        Rect face_i = faces[0];
        // Crop the  face from the image. So simple wuth OpenCV C++
        Mat face = frame_gray(face_i);

        /*
         * prepara de text for insert in the frame
         */
        string box_text;

        // Calculate a position for text
        int pos_x = std::max(face_i.tl().x - 10, 0);
        int pos_y = std::max(face_i.tl().y - 10, 0);


        if (predictionEqual < 10 && predictionNotEqual < 10) {
            // Resizing the face is necessary for Eigenfaces and Fisherfaces
            Mat face_resized;
            resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

            int prediction = model->predict(face_resized);

            if (prediction == faceId) {
                // Draw a green rectangle arround the detected face
                rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
                predictionEqual += 1;
            } else {
                // Draw a green rectangle arround the detected face
                rectangle(original, face_i, CV_RGB(255, 0, 0), 1);
                predictionNotEqual += 1;
            }

            box_text = format("Prediction = %d", prediction);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 255), 2.0);
        }
        else if (predictionEqual >= 10) {
            // Draw a green rectangle arround the detected face
            rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
            box_text = "ACCESS GRANTED";
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
        }
        else {
            // Draw a red rectangle arround the detected face
            rectangle(original, face_i, CV_RGB(255, 0, 0), 1);
            box_text = "ACCESS DENIED";
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
        }
            
        

        imshow("Face Recognizer", original);

        // If press ESC
        char key = (char) waitKey(20);
        if (key == 27) { break; }
    }

    return SUCCESS;
}

int createCSV(int faceId) {
    // path for save the prepared images
    string pathImages = format("%s%d/", path_images.c_str(), faceId);

    // create the folder named faceId, for save the prepared images
    // the name folder is faceId for future indentification
    string command = format("mkdir %s", pathImages.c_str());
    system(command.c_str());

    // file for save the images and respectives labels
    // in format CSV
    string fileName = format("%simages.ext", path_images.c_str());

    // open the file for update and verify
    ofstream file(fileName.c_str(), ios::out | ios::app);

    if (!file) {
        trace(format("(***)ERROR: can't open file %s\n", fileName));
        return FAILURE;
    }

    DIR* folder = opendir(path_temp.c_str());
    if (folder == NULL) {
        trace(format("(!!!)ERROR: the folder %s is NULL\n", path_temp.c_str()));
        return FAILURE;
    }
    struct dirent *ent;

    // read all files in the temp folder
    while ((ent = readdir(folder)) != NULL) {
        char* imageName = ent->d_name;
        // don't write the ocult files in linux
        if (strcmp(imageName, ".") != 0 && strcmp(imageName, "..") != 0) {
            // ; is the separator for CSV file
            string buffer = format("%s%s;%d\n", pathImages.c_str(), imageName, faceId);
            file << buffer;
        }
    }

    // copy all the images in the temp folder, for the images folder
    command = format("cp %s* %s", path_temp.c_str(), pathImages.c_str());
    system(command.c_str());

    // close the file, for elegancy
    file.close();

    return SUCCESS;
}

//////////////////////////////////////////////
// detectAndDisplay
//////////////////////////////////////////////
int detectAndDisplay(Mat frame) {
    vector<Rect> faces;
    vector<Rect> eyes;
    // verify if frame contains 1 face and 2 eyes 
    if (isFace(frame, faces, eyes) == FAILURE)
        return FAILURE;
	
	//convert to gray scale
    Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY );
	if (EQUALIZE)
        equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
	
    // get first face
    // remember, only first face exist in faces vector
    Rect face = faces[0];
	
	// simplify : we only take picture with one face !
	DEBUG printf("(D) detectAndDisplay : nb face=%d\n",faces.size());
	        
    Point center(face.x + face.width / 2, face.y + face.height / 2);
	
	if (eyes_or_glasses == "eyes") {
        trace("-- face without glasses");
	   	// detect eyes
	    for( size_t j = 0; j < 2; j++ ) {
            Point eye_center(face.x + eyes[1-j].x + eyes[1-j].width / 2, face.y + eyes[1-j].y + eyes[1-j].height / 2);

            // left eye
            if (j == 0) {
                Myeye_left.x = eye_center.x;
	         	Myeye_left.y = eye_center.y;
	        }
            // right eye
	        if (j == 1) {
                Myeye_right.x = eye_center.x;
                Myeye_right.y = eye_center.y;
	        }
        }
    } else {
        trace("-- face with glasses");
        
        for( size_t j = 0; j < 2; j++ ) {
            Point eye_center( face.x + eyes[1-j].x + eyes[1-j].width / 2, face.y + eyes[1-j].y + eyes[1-j].height / 2);
            // left eye
            if (j == 0) {
                Myeye_left.x = eye_center.x;
	         	Myeye_left.y = eye_center.y;
            }
            // right eye
            if (j == 1) {
                Myeye_right.x = eye_center.x;
	         	Myeye_right.y = eye_center.y;
            }
        }
    }

	// sometimes eyes are inversed ! we switch them 
	if (Myeye_right.x < Myeye_left.x)
	{
		int tmpX = Myeye_right.x;
		int tmpY = Myeye_right.y;
		Myeye_right.x=Myeye_left.x;
		Myeye_right.y=Myeye_left.y;
		Myeye_left.x=tmpX;
		Myeye_left.y=tmpY;
		trace("-- oups, switch eyes");
		
	}
	
	return SUCCESS;
}

// Detect 1 face and 2 eyes and return SUCCESS
// otherwise return FAILURE
int isFace(Mat frame, vector<Rect>& faces, vector<Rect>& eyes) {
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));

    // if locale face, then locale eyes, else
    // continue the progran for new frame
    if (faces.size() == 1) {
        Mat faceROI = frame_gray(faces[0]); // first frame
        vector<Rect> glasses;

        // In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        glasses_cascade.detectMultiScale(faceROI, glasses, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

        // if find 2 eyes, then prepare and save the image
        if (eyes.size() != 2 && glasses.size() != 2)
            return FAILURE;

        // verify if face contains eyes or glasses
        if (eyes.size() == 2)
            eyes_or_glasses = "eyes";
        else {
            eyes_or_glasses = "glasses";
            eyes = glasses;
        }

        return SUCCESS;
    }

    return FAILURE;
}

///////////////////////////////////////////////////
// trace fonction, output only if #define TRACE=1
///////////////////////////////////////////////////
void trace(string s) {
	if (TRACE==1)
		cout << s <<"\n";
}
 
//////////////////////////////////////////////
// compute distance btw 2 points
//////////////////////////////////////////////
float Distance(CvPoint p1, CvPoint p2) {
	int dx = p2.x - p1.x;
	int dy = p2.y - p1.y;

	return sqrt(dx*dx + dy*dy);
}

 
//////////////////////////////////////////////
// rotate picture (to align eyes-y) 
//////////////////////////////////////////////
Mat rotate(Mat& image, double angle, CvPoint centre) {
    Point2f src_center(centre.x, centre.y);
	// conversion en degre
	angle = angle * 180.0 / 3.14157;
 	DEBUG printf("(D) rotate : rotating : %fÂ° %d %d\n",angle, centre.x, centre.y);
    Mat rot_matrix = getRotationMatrix2D(src_center, angle, 1.0);

    Mat rotated_img(Size(image.size().height, image.size().width), image.type());

    warpAffine(image, rotated_img, rot_matrix, image.size());

    return (rotated_img);
}


 
//////////////////////////////////////////////
// crop picture
//////////////////////////////////////////////
int CropFace(Mat &MyImage, CvPoint eye_left, CvPoint eye_right, CvPoint offset_pct,	CvPoint dest_sz) {
	// calculate offsets in original image
	int offset_h = (offset_pct.x * dest_sz.x / 100);
	int offset_v = (offset_pct.y * dest_sz.y / 100);
	DEBUG printf("(D) CropFace : offset_h=%d, offset_v=%d\n",offset_h,offset_v);
	
	// get the direction
	CvPoint eye_direction;
	eye_direction.x = eye_right.x - eye_left.x;
	eye_direction.y = eye_right.y - eye_left.y;
	
	
	// calc rotation angle in radians
	float rotation = atan2((float)(eye_direction.y), (float)(eye_direction.x));
	
	// distance between them
	float dist = Distance(eye_left, eye_right);
	DEBUG printf("(D) CropFace : dist=%f\n",dist);
	
	// calculate the reference eye-width
	int reference = dest_sz.x - 2 * offset_h;
	
	// scale factor
	float scale = dist / (float)reference;
    DEBUG printf("(D) CropFace : scale=%f\n",scale);
	
	// rotate original around the left eye
	char sTmp[16];
	sprintf(sTmp,"%f",rotation);
	trace("-- rotate image "+string(sTmp));
	MyImage = rotate(MyImage, (double)rotation, eye_left); 
	
	// crop the rotated image
	CvPoint crop_xy;
	crop_xy.x = eye_left.x - scale * offset_h;
	crop_xy.y = eye_left.y - scale * offset_v;
	
	CvPoint crop_size;
	crop_size.x = dest_sz.x * scale; 
	crop_size.y = dest_sz.y * scale;
	
	// Crop the full image to that image contained by the rectangle myROI
	trace("-- crop image");
	DEBUG printf("(D) CropFace : crop_xy.x=%d, crop_xy.y=%d, crop_size.x=%d, crop_size.y=%d",crop_xy.x, crop_xy.y, crop_size.x, crop_size.y);
	
	Rect myROI(crop_xy.x, crop_xy.y, crop_size.x, crop_size.y);
	if ((crop_xy.x+crop_size.x<MyImage.size().width)&&(crop_xy.y+crop_size.y<MyImage.size().height))
        MyImage = MyImage(myROI);
	else {
			trace("-- error cropping");
			return FAILURE;
	}
  
    //resize it
    trace("-- resize image");
    resize(MyImage, MyImage, Size(dest_sz));
  
    return SUCCESS;
}

 
//////////////////////////////////////////////
// resize picture
//////////////////////////////////////////////
void resizePicture(Mat& src, int coeff)  {
	// Resize src to img size 
	Size oldTaille = src.size(); 
	Size newTaille(coeff,oldTaille.height*coeff/oldTaille.width);
	cv::resize(src, src, newTaille); 
}

void readImageFolder(const String& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    cout << "try open file: " << filename << endl;
    std::ifstream file(filename.c_str(), ifstream::in);

    if (!file) {
        string error_message = "No valid input file";
        CV_Error(CV_StsBadArg, error_message);
    }

    cout << "file is open" << endl;

    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);

        if (!path.empty() && !classlabel.empty()) {
            cout << path << endl;
            cout << atoi(classlabel.c_str()) << endl;
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    file.close();
}

int loadCascades() {
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    if (!eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading eyes cascade\n");
        return -1;
    }

    if (!glasses_cascade.load(glasses_cascade_name)) {
        printf("--(!)Error loading glasses cascade\n");
        return -1;
    }

    return 0;
}

int  getLastLabel() {
    printf("--- INIT getLastLabel() ---\n"); // LOG
    ifstream file;
    string filename("images/image.ext");

    if (!ifstream(filename)) {
        printf("(***)File: %s not exist, returning 0\n\n", filename.c_str()); // LOG
        printf("--- END getLastLabe() ---\n\n"); // LOG
        return 0; 
    }

    printf("(***)trying open file %s\n", filename.c_str()); // LOG
    file.open(filename, ios::in);

    if (!file) {
        CV_Error(CV_StsBadArg, "Error when opening: <images/image.ext>");
        printf("--- END getLastLabe() ---\n\n"); // LOG
        return -1;
    }
    printf("(***)OK SUCESS!!!\n\n"); // LOG

    string line, last_line;

    while (getline(file, line)) { last_line = line; }

    line = last_line.back();
    int last_label = atoi(line.c_str());

    // Only for fun
    file.close();

    printf("(***)Returning %d\n", last_label + 1); // LOG
    printf("--- END getLastLabel() ---\n\n"); // LOG

    return (last_label + 1);
}

void saveFaceFrame(Mat frame, int num_img, int label) {
    printf("---INIT saveFaceFrame() ---\n");

    char path_and_label[50], path_image_frame[50], folder[5];
    string filename("images/image.ext");
    string buffer, command;
    ofstream file_ext;

    if (num_img == 0 && label == 0) {
        file_ext.open(filename, ios::out);
        sprintf(path_and_label, "images/%d/image%d.jpg;%d", label, num_img, label);
    } else {
        file_ext.open(filename, ios::out | ios::app);
        sprintf(path_and_label, "\nimages/%d/image%d.jpg;%d", label, num_img, label);
    }

    if (!file_ext) {
        CV_Error(CV_StsBadArg, "Error when opening: <images/image.ext>");
        printf("--- END saveFaceFrame() ---\n\n");
        exit(1);
    }

    printf("(***)FILE %s IS OPEN:\n", filename.c_str());
    
    // save de path to images in the file *.ext
    buffer = path_and_label;
    printf("(***)try write %s in %s\n", buffer.c_str(), filename.c_str());
    cout << ">>>> buffer is: " << buffer << endl;
    cout << ">>>> buffer.c_str() is: " << buffer.c_str() << endl;
    file_ext << buffer.c_str();
    buffer.clear();
    // End of process, close the file for elegancy
    file_ext.close();
    
    printf("(***)CLOSE FILE <%s>\n", filename.c_str());
    printf("(***)OK SUCESS!!!\n\n");

    // verify and create folder when them not exists
    // label is too a folder name for the images
    sprintf(folder, "images/%d", label);
    if (stat(folder, &sb) != 0 && !S_ISDIR(sb.st_mode)) {
        command = "mkdir ";
        command += folder;
        printf("(***)EXECUTING COMMAND: %s\n", command.c_str());
        system(command.c_str());
        printf("(***)OK SUCESS!!!\n\n");
    }

    // save the image/frame in the correct path
    sprintf(path_image_frame, "images/%d/image%d.jpg", label, num_img);
    buffer = path_image_frame;

    printf("(***)WRITING IMAGE/FRAME IN: %s\n", buffer.c_str());
    imwrite(buffer.c_str(), frame);
    buffer.clear();
    printf("(***)OK SUCESS!!!\n\n");

    printf("--- END saveFaceFrame() ---\n\n");
}







