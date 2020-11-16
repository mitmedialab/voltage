#!/bin/bash
function build_clean {
	rm -fr fastCoordDescent.cp* fastCoordDescent.egg-info/; make clean;	
}

function error_msg {
	echo "Incorrect command!"
	echo "Usage:"
	echo "bash build.sh (OPTIONAL)[clean]"
	exit
}

if [ $# -eq 1 ]; then
	if [ $1 = "clean" ]; then
		build_clean
	else
		error_msg
	fi
else
	build_clean
	make
fi
		
