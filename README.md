# VA3DR

This repository contains the code used in the image matching algorithm for the project "Visual Autonomy in 3D Renderings."

ICE.py contains a compilation of image matching algorithms that work together to quickly and accurately match images with telemetry.

Instructions on instaling the necessary libraries and running the code can be found in ICEInstallationGuideUbuntu.txt

## ICE Installation Guide

Make sure you have Python 2.7.X (the X can be anything) and Sublime
(http://askubuntu.com/questions/521265/how-to-install-sublime-text-2-instead-of-3-from-webupd8team-ppa)

Install the necessary libraries to the computer (not to a directory within the computer).
OpenCV: http://rodrigoberriel.com/2014/10/installing-opencv-3-0-0-on-ubuntu-14-04/
Scipy: http://www.scipy.org/install.html
Scikit-Image: http://scikit-image.org/download.html
Download the necessary Python files from the Code folder - pay attention to the correct version of the code that corresponds to the version of openCV you have.

To check that everything works, run round1.py. If it runs free of errors, you’re good to do!

You need to make sure you have all your images in the same directory, and that the name of this directory is put into the proper place in the round1.py file.
This first set of images must all be the same size.
You also need a second directory for the smaller images. Again, place this directory name into the round1.py file at the proper location.
It needs to be named 200images

OpenCV
1st step: Install the Dependencies
To install the dependencies required from OpenCV, just run the following commands:

sudo apt-get -y install libopencv-dev build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff4-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
2nd step: Download OpenCV 3.0.0 alpha
You can download manually or run the commands below to get OpenCV:

mkdir opencv
cd opencv
wget https://github.com/Itseez/opencv/archive/3.0.0-alpha.zip -O opencv-3.0.0-alpha.zip
unzip opencv-3.0.0-alpha.zip
3rd step: Install OpenCV
Now, we’ll install OpenCV. Cmake command has a lot of options: choose those that better suit your needs and run the commands below. If you’re planning to use Qt 5.x, don’t use -D WITH_QT=ON. Learn how to use OpenCV 3 with Qt Creator 3.2 (Qt 5.3). According to one of the users that tested it on Ubuntu 14.10, you’ll need to use WITH_FFMPEG=OFF.

cd opencv-3.0.0-alpha
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j $(nproc)
sudo make install
4rd step: Finishing installation
To get OpenCV working properly, we need to tell Ubuntu:

sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
