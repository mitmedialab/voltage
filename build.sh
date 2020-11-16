#!/bin/bash 

LOG_FILE="${PWD}/log_build.txt"

function build_clean_all {
	echo >> ${LOG_FILE}
	echo "Cleaning Preprocessing..." | tee -a ${LOG_FILE}
	cd preprocessing
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
	echo >> ${LOG_FILE}
	echo "Cleaning fastCoordDescent..." | tee -a ${LOG_FILE}
	cd fastCoordDescent
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
	echo >> ${LOG_FILE}
	echo "Cleaning Postprocessing..." | tee -a ${LOG_FILE}
	cd postprocessing
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
}


function clean_preprocessing {
	echo >> ${LOG_FILE}
	echo "Cleaning Preprocessing..." | tee -a ${LOG_FILE}
	cd preprocessing
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
}

function clean_fastCoordDescent {
	echo >> ${LOG_FILE}
	echo "Cleaning fastCoordDescent..." | tee -a ${LOG_FILE}
	cd fastCoordDescent
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
}

function clean_postprocessing {
	echo >> ${LOG_FILE}
	echo "Cleaning Postprocessing..." | tee -a ${LOG_FILE}
	cd postprocessing
	bash build.sh clean  >> ${LOG_FILE}
	cd ..
}


function build_all {
	echo >> ${LOG_FILE}
	echo "Building Preprocessing..." | tee -a ${LOG_FILE}
	cd preprocessing
	bash build.sh  >> ${LOG_FILE}
	cd ..
	echo >> ${LOG_FILE}
	echo "Building fastCoordDescent..." | tee -a ${LOG_FILE}
	cd fastCoordDescent
	bash build.sh  >> ${LOG_FILE}
	cd ..
	echo >> ${LOG_FILE}
	echo "Building Postprocessing..." | tee -a ${LOG_FILE}
	cd postprocessing
	bash build.sh  >> ${LOG_FILE}
	cd ..
}


function build_preprocessing {
	echo >> ${LOG_FILE}
	echo "Building Preprocessing..." | tee -a ${LOG_FILE}
	cd preprocessing
	bash build.sh  >> ${LOG_FILE}
	cd ..
}

function build_fastCoordDescent {
	echo >> ${LOG_FILE}
	echo "Building fastCoordDescent..." | tee -a ${LOG_FILE}
	cd fastCoordDescent
	bash build.sh  >> ${LOG_FILE}
	cd ..
}

function build_postprocessing {
	echo >> ${LOG_FILE}
	echo "Building Postprocessing..." | tee -a ${LOG_FILE}
	cd postprocessing
	bash build.sh  >> ${LOG_FILE}
	cd ..
}


function error_msg {
	echo >> ${LOG_FILE}
	echo "Incorrect command!"
	echo "Usage:"
	echo "bash build.sh (OPTIONAL)[clean]"
	exit
}
echo >> ${LOG_FILE}
echo >> ${LOG_FILE}
echo "Initiating build script" >> ${LOG_FILE}
date >> ${LOG_FILE}
echo >> ${LOG_FILE}

if [ $# -eq 1 ]; then
	if [ $1 = "clean" ]; then
		build_clean_all
		rm -fr __pycache__
	elif [ $1 = "preprocessing" ]; then
		build_preprocessing
	elif [ $1 = "fastCoordDescent" ]; then
		build_fastCoordDescent
	elif [ $1 = "postprocessing" ]; then
		build_postprocessing
	else
		error_msg
	fi
elif [ $# -eq 2 ]; then
	if [ $1 = 'preprocessing' ] && [ $2 = 'clean' ]; then
		clean_preprocessing
	elif [ $1 = 'fastCoordDescent' ] && [ $2 = 'clean' ]; then
		clean_fastCoordDescent
	elif [ $1 = 'postprocessing' ] && [ $2 = 'clean' ]; then
		clean_postprocessing
	else
		error_msg
	fi
elif [ $# -gt 2 ]; then
	error_msg
else
	build_all
fi
		
